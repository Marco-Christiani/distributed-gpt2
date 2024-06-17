const std = @import("std");
const net = std.net;
const ops = @import("ops.zig");
const bpe = @import("bpe.zig");

// pub const std_options = struct {
//     pub const log_level = .info;
// };

const Models = struct {
    const Small = GPTConfig{ .name = @constCast("124M"), .vocab_size = 50257, .context_size = 1024, .n_layer = 12, .n_heads = 12, .n_embed = 768 };
    const Medium = GPTConfig{ .name = @constCast("355M"), .vocab_size = 50257, .context_size = 1024, .n_layer = 24, .n_heads = 16, .n_embed = 1024 };
    const Large = GPTConfig{ .name = @constCast("774M"), .vocab_size = 50257, .context_size = 1024, .n_layer = 36, .n_heads = 20, .n_embed = 1280 };
    const XL = GPTConfig{ .name = @constCast("1558M"), .vocab_size = 50257, .context_size = 1600, .n_layer = 48, .n_heads = 25, .n_embed = 1600 };
};

const modelConfig = Models.XL;

const GPTConfig = struct {
    const Self = @This();

    name: []u8,
    vocab_size: usize,
    context_size: usize,
    n_layer: usize,
    n_heads: usize,
    n_embed: usize,

    pub fn init(name: []u8, vocab_size: usize, context_size: usize, n_layer: usize, n_heads: usize, n_embed: usize) Self {
        return Self{
            .name = name,
            .vocab_size = vocab_size,
            .context_size = context_size,
            .n_layer = n_layer,
            .n_heads = n_heads,
            .n_embed = n_embed,
        };
    }
};

const GPTShardConfig = struct {
    rank: usize,
    serverPort: u16,
    nextAddress: net.Address,
    startBlock: usize,
    endBlock: usize,
};

/// Structure which maintains state which is shared across all GPT layers.
pub const State = struct {
    const Self = @This();

    pos_emb: []f32,
    x: []f32,
    o: []f32,
    logits: []f32,
    decoded: []u8,

    // Intermediate buffers.
    _h: []f32,
    _4xh: []f32,
    _qkv: []f32,
    _q: []f32,
    _k: []f32,
    _v: []f32,
    _attn: []f32,

    allocator: std.mem.Allocator,

    pub fn init(config: GPTConfig, allocator: std.mem.Allocator) !Self {
        return Self{
            .pos_emb = try allocator.alloc(f32, 1 * config.n_embed),
            .x = try allocator.alloc(f32, 1 * config.n_embed),
            .o = try allocator.alloc(f32, 1 * config.n_embed),
            .logits = try allocator.alloc(f32, config.vocab_size),
            .decoded = try allocator.alloc(u8, 20),

            ._h = try allocator.alloc(f32, 1 * config.n_embed),
            ._4xh = try allocator.alloc(f32, 1 * 4 * config.n_embed),
            ._qkv = try allocator.alloc(f32, 1 * 3 * config.n_embed),
            ._q = try allocator.alloc(f32, 1 * config.n_embed),
            ._k = try allocator.alloc(f32, config.context_size * config.n_embed),
            ._v = try allocator.alloc(f32, config.context_size * config.n_embed),
            ._attn = try allocator.alloc(f32, 1 * config.context_size),

            .allocator = allocator,
        };
    }
    pub fn totalMemoryUsage(self: State) usize {
        const posEmbSize = @sizeOf(f32) * self.pos_emb.len;
        const xSize = @sizeOf(f32) * self.x.len;
        const oSize = @sizeOf(f32) * self.o.len;
        const logitsSize = @sizeOf(f32) * self.logits.len;
        const decodedSize = @sizeOf(u8) * self.decoded.len;

        const hSize = @sizeOf(f32) * self._h.len;
        const _4xhSize = @sizeOf(f32) * self._4xh.len;
        const qkvSize = @sizeOf(f32) * self._qkv.len;
        const qSize = @sizeOf(f32) * self._q.len;
        const kSize = @sizeOf(f32) * self._k.len;
        const vSize = @sizeOf(f32) * self._v.len;
        const attnSize = @sizeOf(f32) * self._attn.len;

        const baseSize = @sizeOf(@This());

        return baseSize + posEmbSize + xSize + oSize + logitsSize + decodedSize +
            hSize + _4xhSize + qkvSize + qSize + kSize + vSize + attnSize;
    }
};

const MLP = struct {
    const Self = @This();

    c_fc: ops.Linear,
    c_proj: ops.Linear,

    pub fn init(c_fc: ops.Linear, c_proj: ops.Linear) MLP {
        return MLP{ .c_fc = c_fc, .c_proj = c_proj };
    }

    /// Computes the forward pass and writes the result to state.o.
    pub fn forward(self: Self, inputs: []const f32, state: State) void {
        self.c_fc.forward(inputs, state._4xh);
        ops.gelu(state._4xh);
        self.c_proj.forward(state._4xh, state.o);
    }
};

const Block = struct {
    const Self = @This();

    n_embed: usize,
    ln_1: ops.LayerNorm,
    attn: ops.CausalSelfAttention,
    ln_2: ops.LayerNorm,
    mlp: MLP,
    k_cache: []f32,
    v_cache: []f32,

    pub fn init(
        n_embed: usize,
        ln_1: ops.LayerNorm,
        attn: ops.CausalSelfAttention,
        ln_2: ops.LayerNorm,
        mlp: MLP,
        k_cache: []f32,
        v_cache: []f32,
    ) Self {
        return Self{
            .n_embed = n_embed,
            .ln_1 = ln_1,
            .attn = attn,
            .ln_2 = ln_2,
            .mlp = mlp,
            .k_cache = k_cache,
            .v_cache = v_cache,
        };
    }

    /// Computes the forward pass and writes the result to both state.x and state.o. This
    /// enables sequentially calling multiple Block.forwards() in a row without having to copy
    /// memory.
    pub fn forward(self: Self, seq_len: usize, inputs: []const f32, state: State) void {
        // Create a copy of x for residual computation.
        @memcpy(state._h, inputs);

        self.ln_1.forward(state._h);
        self.attn.forward(
            seq_len,
            state._h,
            self.k_cache[0 .. seq_len * self.n_embed],
            self.v_cache[0 .. seq_len * self.n_embed],
            state.o,
            state._qkv,
            state._q,
            state._k[0 .. seq_len * self.n_embed],
            state._v[0 .. seq_len * self.n_embed],
            state._attn[0..seq_len],
        );
        for (0..state.o.len) |i| {
            state._h[i] = state.o[i] + inputs[i];
            state.x[i] = state._h[i];
        }
        self.ln_2.forward(state._h);
        self.mlp.forward(state._h, state);
        for (0..state.o.len) |i| {
            state.o[i] += state.x[i];
            state.x[i] = state.o[i];
        }
    }
};

const GPTShard = struct {
    const Self = @This();

    config: GPTConfig,
    shardConfig: GPTShardConfig,
    wte: ?ops.Embedding,
    wpe: ?ops.Embedding,
    h: []const Block,
    ln_f: ?ops.LayerNorm,
    lm_head: ?ops.Linear,
    server: *net.StreamServer,
    connection: ?net.StreamServer.Connection,
    nextStream: ?net.Stream,

    pub fn startServer(self: *Self) !void {
        std.debug.print("{} Listening on {}\n", .{ self.shardConfig.rank, self.shardConfig.serverPort });
        try self.server.listen(try net.Address.resolveIp("0.0.0.0", self.shardConfig.serverPort));
    }

    pub fn acceptConnection(self: *Self) void {
        std.debug.print("{} Waiting for connection on {}\n", .{ self.shardConfig.rank, self.shardConfig.serverPort });
        while (true) {
            // for zig 11
            // self.connection = self.server.accept() catch @panic("accept failed!");
            if (self.server.sockfd == null) {
                std.debug.print("{} sockfd is null sleeping\n", .{self.shardConfig.rank});
                std.time.sleep(1000 * std.time.ns_per_ms);
                continue;
            }
            self.connection = self.server.accept() catch |err| {
                std.debug.print("Accept failed: {}\n", .{err});
                std.debug.print("{} Accept retrying...\n", .{self.shardConfig.rank});
                std.time.sleep(1000 * std.time.ns_per_ms);
                continue;
            };
            // for zig 12
            // self.connection = self.server.accept() catch |err| switch (err) {
            //     error.WouldBlock => {
            //         std.debug.print("{} Accept retrying...\n", .{self.shardConfig.rank});
            //         std.time.sleep(1000 * std.time.ns_per_ms);
            //         continue;
            //     },
            //     else => {
            //         std.debug.panic("Accept failed: {}", .{err});
            //     },
            // };
            std.debug.print("{} Connection accepted from {}\n", .{ self.shardConfig.rank, self.connection.?.address });
            break;
        }
    }

    pub fn connect(self: *Self) void {
        std.debug.print("{} Connecting to {}\n", .{ self.shardConfig.rank, self.shardConfig.nextAddress });
        while (true) {
            // self.nextStream = net.tcpConnectToAddress(self.shardConfig.nextAddress) catch @panic("connect failed");
            self.nextStream = net.tcpConnectToAddress(self.shardConfig.nextAddress) catch |err| {
                std.debug.print("Connect failed: {}\n", .{err});
                std.debug.print("  {} retrying retrying...\n", .{self.shardConfig.rank});
                std.time.sleep(1000 * std.time.ns_per_ms);
                continue;
            };
            // self.nextStream = net.tcpConnectToAddress(self.shardConfig.nextAddress) catch |err| switch (err) {
            //     error.WouldBlock => {
            //         std.debug.print("{} connect() retrying..\n", .{self.shardConfig.rank});
            //         std.time.sleep(1000 * std.time.ns_per_ms);
            //         continue;
            //     },
            //     else => {
            //         std.debug.panic("Connection failed due to socket error: {} ", .{err});
            //     },
            // };
            std.debug.print("{} Connected {?any}\n", .{ self.shardConfig.rank, self.nextStream });
            break;
        }
    }

    pub fn load(config: GPTConfig, shardConfig: GPTShardConfig, allocator: std.mem.Allocator) !Self {
        const layersInShard = shardConfig.endBlock - shardConfig.startBlock;
        var h = try allocator.alloc(Block, layersInShard);
        for (0..h.len) |i| {
            std.debug.print("Loading h[{}]\n", .{shardConfig.startBlock + i});
            h[i] = try load_block(shardConfig.startBlock + i, config, allocator);
        }
        var self: Self = Self{
            .config = config,
            .shardConfig = shardConfig,
            .wte = null,
            .wpe = null,
            .h = h,
            .ln_f = null,
            .lm_head = null,
            .server = @constCast(&net.StreamServer.init(.{ .reuse_address = true, .reuse_port = true })),
            .connection = null,
            .nextStream = null,
        };
        const wte = try load_embedding("wte", config.vocab_size, config.n_embed, allocator);
        if (shardConfig.startBlock == 0) {
            self.wte = wte;
            self.wpe = try load_embedding("wpe", config.context_size, config.n_embed, allocator);
        }
        if (shardConfig.endBlock == config.n_layer) {
            self.ln_f = try load_layer_norm("ln_f", config.n_embed, allocator);
            self.lm_head = ops.Linear.init(config.n_embed, config.vocab_size, wte.weight, null);
        }
        return self;
    }

    pub fn recvState(self: *Self, state: State) void {
        var rcvBuf: [modelConfig.n_embed * (@sizeOf(f32) / @sizeOf(u8))]u8 = undefined;
        if (self.connection) |conn| {
            const rcvLen = conn.stream.readAll(&rcvBuf) catch |err| blk: {
                std.log.err("{} read failed {}\n", .{ self.shardConfig.rank, err });
                break :blk 0;
            };
            std.log.debug("{} read {} bytes\n", .{ self.shardConfig.rank, rcvLen });
            @memcpy(state.x, std.mem.bytesAsSlice(f32, rcvBuf[0..rcvLen]));
            // @memcpy(state.o, std.mem.bytesAsSlice(f32, rcvBuf[0..rcvLen]));
        } else {
            std.log.debug("{} no connection\n", .{self.shardConfig.rank});
        }
    }
    pub fn sendState(self: *Self, state: State) void {
        if (self.nextStream) |stream| {
            const sendBuf = std.mem.sliceAsBytes(state.o);
            stream.writeAll(sendBuf) catch @panic("Failed to send bytes");
        } else {
            std.log.debug("{} No nextStream\n", .{self.shardConfig.rank});
        }
    }

    // Computes the forward pass and writes the result in state.logits.
    pub fn forward(self: Self, seq_len: usize, token: usize, state: State) void {
        if (self.shardConfig.startBlock == 0) self.firstForward(seq_len, token, state);
        self.middleForward(seq_len, state);
        if (self.shardConfig.endBlock == self.config.n_layer) self.lastForward(state);
    }

    pub fn firstForward(self: Self, seq_len: usize, token: usize, state: State) void {
        self.wpe.?.forward(&[1]usize{seq_len - 1}, state.pos_emb);
        self.wte.?.forward(&[1]usize{token}, state.x);
        for (0..self.config.n_embed) |i| {
            state.x[i] += state.pos_emb[i];
        }
    }
    pub fn middleForward(self: Self, seq_len: usize, state: State) void {
        for (0..self.h.len) |i| {
            self.h[i].forward(seq_len, state.x, state);
        }
    }

    pub fn lastForward(self: Self, state: State) void {
        self.ln_f.?.forward(state.x);
        self.lm_head.?.forward(state.x, state.logits);
    }

    /// Samples the next token.
    pub fn sample(self: Self, seq_len: usize, temp: f32, token: u32, state: State) u32 {
        self.forward(seq_len, token, state);
        return self._sample_logits(temp, state);
    }
    pub fn _sample_logits(self: Self, temp: f32, state: State) u32 {
        _ = self;
        for (0..state.logits.len) |i| {
            state.logits[i] /= temp;
        }
        ops.softmax(state.logits);
        var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
        var random = rng.random();
        return @intCast(random.weightedIndex(f32, state.logits));
    }
};

pub fn load_linear(
    name: []const u8,
    in_features: usize,
    out_features: usize,
    allocator: std.mem.Allocator,
) !ops.Linear {
    const weight_path = try std.fmt.allocPrint(allocator, "models/{s}/raw/model-{s}-w", .{ modelConfig.name, name });
    defer allocator.free(weight_path);
    const weight = try ops.load_tensor(
        weight_path,
        &[_]usize{ in_features, out_features },
        f32,
        allocator,
    );
    const bias_path = try std.fmt.allocPrint(allocator, "models/{s}/raw/model-{s}-b", .{ modelConfig.name, name });
    defer allocator.free(bias_path);
    const bias = try ops.load_tensor(
        bias_path,
        &[_]usize{out_features},
        f32,
        allocator,
    );
    return ops.Linear.init(in_features, out_features, weight, bias);
}

pub fn load_layer_norm(
    name: []const u8,
    n_features: usize,
    allocator: std.mem.Allocator,
) !ops.LayerNorm {
    const weight_path = try std.fmt.allocPrint(allocator, "models/{s}/raw/model-{s}-g", .{ modelConfig.name, name });
    defer allocator.free(weight_path);
    const weight = try ops.load_tensor(
        weight_path,
        &[_]usize{n_features},
        f32,
        allocator,
    );
    const bias_path = try std.fmt.allocPrint(allocator, "models/{s}/raw/model-{s}-b", .{ modelConfig.name, name });
    defer allocator.free(bias_path);
    const bias = try ops.load_tensor(
        bias_path,
        &[_]usize{n_features},
        f32,
        allocator,
    );
    return ops.LayerNorm.init(n_features, weight, bias);
}

pub fn load_embedding(name: []const u8, vocab_size: usize, emb_dim: usize, allocator: std.mem.Allocator) !ops.Embedding {
    const path = try std.fmt.allocPrint(allocator, "models/{s}/raw/model-{s}", .{ modelConfig.name, name });
    defer allocator.free(path);
    const weight = try ops.load_tensor(
        path,
        &[_]usize{ vocab_size, emb_dim },
        f32,
        allocator,
    );
    return ops.Embedding.init(emb_dim, weight);
}

pub fn load_block(layer_idx: usize, config: GPTConfig, allocator: std.mem.Allocator) !Block {
    const ln_1_name = try std.fmt.allocPrint(allocator, "h{any}-ln_1", .{layer_idx});
    defer allocator.free(ln_1_name);
    const ln_1 = try load_layer_norm(ln_1_name, config.n_embed, allocator);

    const c_attn_name = try std.fmt.allocPrint(allocator, "h{any}-attn-c_attn", .{layer_idx});
    defer allocator.free(c_attn_name);
    const c_attn = try load_linear(c_attn_name, config.n_embed, 3 * config.n_embed, allocator);

    const c_proj_name = try std.fmt.allocPrint(allocator, "h{any}-attn-c_proj", .{layer_idx});
    defer allocator.free(c_proj_name);
    const c_proj = try load_linear(c_proj_name, config.n_embed, config.n_embed, allocator);

    const ln_2_name = try std.fmt.allocPrint(allocator, "h{any}-ln_2", .{layer_idx});
    defer allocator.free(ln_2_name);
    const ln_2 = try load_layer_norm(ln_2_name, config.n_embed, allocator);

    const c_fc_name = try std.fmt.allocPrint(allocator, "h{any}-mlp-c_fc", .{layer_idx});
    defer allocator.free(c_fc_name);
    const c_fc = try load_linear(c_fc_name, config.n_embed, 4 * config.n_embed, allocator);

    const mlp_c_proj_name = try std.fmt.allocPrint(allocator, "h{any}-mlp-c_proj", .{layer_idx});
    defer allocator.free(mlp_c_proj_name);
    const mlp_c_proj = try load_linear(mlp_c_proj_name, 4 * config.n_embed, config.n_embed, allocator);

    const attn = ops.CausalSelfAttention.init(config.n_heads, config.n_embed, c_attn, c_proj);
    const mlp = MLP.init(c_fc, mlp_c_proj);
    const k_cache = try allocator.alloc(f32, config.context_size * config.n_embed);
    const v_cache = try allocator.alloc(f32, config.context_size * config.n_embed);

    return Block.init(config.n_embed, ln_1, attn, ln_2, mlp, k_cache, v_cache);
}

// pub fn load_gpt(config: GPTConfig, allocator: std.mem.Allocator) !GPT {
//     const wte = try load_embedding("wte", config.vocab_size, config.n_embed, allocator);
//     const wpe = try load_embedding("wpe", config.context_size, config.n_embed, allocator);
//     var h = try allocator.alloc(Block, config.n_layer);
//     for (0..h.len) |i| {
//         h[i] = try load_block(i, config, allocator);
//     }
//     const ln_f = try load_layer_norm("ln_f", config.n_embed, allocator);
//     const lm_head = ops.Linear.init(config.n_embed, config.vocab_size, wte.weight, null);
//     return GPT.init(config, wte, wpe, h, ln_f, lm_head);
// }

pub fn load_encoder(allocator: std.mem.Allocator) !bpe.Encoder {
    const encoder_path = try std.fmt.allocPrint(allocator, "models/{s}/encoder.json", .{modelConfig.name});
    defer allocator.free(encoder_path);
    const parsed_encoder = try ops.load_json(encoder_path, allocator);

    const byte_encoder_path = try std.fmt.allocPrint(allocator, "models/{s}/byte_encoder.json", .{modelConfig.name});
    defer allocator.free(byte_encoder_path);
    const parsed_bytes_encoder = try ops.load_json(byte_encoder_path, allocator);
    return bpe.Encoder.init(parsed_encoder.object, parsed_bytes_encoder.object, allocator);
}

pub fn generateShardedTcpDistributed(
    gptShard_: GPTShard,
    encoder: bpe.Encoder,
    temp: f32,
    inputs: []u32,
    state: State,
) !void {
    var gptShard = @constCast(&gptShard_);

    defer {
        std.debug.print("Closing stream\n", .{});
        if (gptShard.connection) |conn| conn.stream.close();
        // std.debug.print("Stopping server {}\n", .{}); // couldnt tell ya why the fd is bad...
        // gptShard.server.deinit();
    }
    if (gptShard.shardConfig.rank == 1) {
        std.time.sleep(2 * std.time.ns_per_s);
        gptShard.connect();
        try gptShard.startServer();
        gptShard.acceptConnection();
        var rcvBuf: [10]u8 = undefined;
        _ = try gptShard.connection.?.stream.read(&rcvBuf);
        std.debug.print("Received {s}\n", .{rcvBuf});
    } else {
        try gptShard.startServer();
        gptShard.acceptConnection();
        gptShard.connect();
        _ = try gptShard.nextStream.?.writeAll("Ping!");
        std.debug.print("Sent ping\n", .{});
    }

    var token: u32 = inputs[0];
    var pred: u32 = undefined;
    var seq_len: usize = 0;

    while (seq_len < gptShard.config.context_size) {
        // first shard starts generation
        if (gptShard.shardConfig.rank == 0) {
            if (seq_len > 0) {
                // recv sampled token
                var rcvBuf: [4]u8 = undefined;
                if (gptShard.connection) |conn| {
                    std.log.info("\nReceiving token {} bytes\n", .{rcvBuf.len});
                    const rcvLen = conn.stream.readAll(&rcvBuf) catch |err| blk: {
                        std.log.err("{} read failed {}\n", .{ gptShard.shardConfig.rank, err });
                        break :blk 0;
                    };
                    pred = std.mem.bytesAsSlice(u32, rcvBuf[0..rcvLen])[0];
                }
                gptShard.recvState(state);
            }
            if (seq_len >= inputs.len) {
                token = pred;
            } else {
                token = inputs[seq_len];
            }
            const decoded_len = encoder.decode(&[_]u32{token}, state.decoded);
            std.debug.print("{s}", .{state.decoded[0..decoded_len]});
            gptShard.forward(seq_len + 1, token, state);
            gptShard.sendState(state);
            seq_len += 1;
        } // last shard samples and sends token back to first
        else if (gptShard.shardConfig.endBlock == gptShard.config.n_layer) {
            gptShard.recvState(state);
            gptShard.middleForward(seq_len + 1, state);
            gptShard.lastForward(state);
            // sample and send token to first shard
            token = gptShard._sample_logits(temp, state);
            const decoded_len = encoder.decode(&[_]u32{token}, state.decoded);
            std.debug.print("{s}", .{state.decoded[0..decoded_len]});

            if (gptShard.nextStream) |stream| {
                // send token to first shard
                const sendBuf = std.mem.sliceAsBytes(&[_]@TypeOf(token){token});
                // const sendBuf = std.mem.asBytes(&std.mem.nativeToBig(u32, token));
                std.log.info("\nSending token: \"{s}\"\nDec: {}\nHex: 0x{x}\nBytes: {}\n", .{ state.decoded[0..decoded_len], token, token, sendBuf.len });
                std.log.info("Type: {}\n", .{@TypeOf(sendBuf)});
                stream.writeAll(sendBuf) catch @panic("Failed to send bytes");
                // send state to first shard
                gptShard.sendState(state);
            } else {
                std.log.err("No connection, likely closed.\n", .{});
                return;
            }
            seq_len += 1;
        } else {
            @panic("Middle not implemented yet.");
        }
    }
    // const decoded_len = encoder.decode(&[_]usize{token}, state.decoded);
    // std.debug.print("{s}", .{state.decoded[0..decoded_len]});
}

pub fn generateShardedLocal(
    gptShards: []const GPTShard,
    encoder: bpe.Encoder,
    temp: f32,
    inputs: []usize,
    state: State,
) void {
    var token: usize = undefined;
    for (0..gptShards[0].config.context_size) |s| {
        if (s < inputs.len) {
            token = inputs[s];
        }
        // fwd pass on first n-1 shards
        for (0..gptShards.len - 1) |i| {
            const gpt = gptShards[i];
            gpt.forward(s + 1, token, state);
        }
        // last shard either samples a token or also fwd pass
        const gpt = gptShards[gptShards.len - 1];
        if (s >= inputs.len) {
            token = gpt.sample(s + 1, temp, token, state);
        } else {
            gpt.forward(s + 1, token, state);
        }
        const decoded_len = encoder.decode(&[_]usize{token}, state.decoded);
        std.debug.print("{s}", .{state.decoded[0..decoded_len]});
    }
}

pub fn generateShardedTcpLocal(
    gptShards: []const GPTShard,
    encoder: bpe.Encoder,
    temp: f32,
    inputs: []usize,
    state: State,
) !void {
    defer {
        for (0..gptShards.len) |i| {
            std.debug.print("Closing stream {}\n", .{i});
            if (gptShards[i].nextStream) |stream| stream.close();
            // std.debug.print("Stopping server {}\n", .{i}); // couldnt tell ya why the fd is bad...
            // gptShards[i].server.deinit();
        }
    }

    for (0..gptShards.len) |i| {
        const next = (i + 1) % gptShards.len;
        try @constCast(&gptShards[next]).startServer();
        @constCast(&gptShards[i]).connect();
        @constCast(&gptShards[next]).acceptConnection();
        std.debug.print("Boop!\n", .{});
    }
    var token: usize = undefined;
    for (0..gptShards[0].config.context_size) |s| {
        if (s < inputs.len) {
            token = inputs[s];
        }
        // fwd pass on first n-1 shards
        for (0..gptShards.len - 1) |i| {
            const gpt = gptShards[i];
            gpt.forward(s + 1, token, state);
        }
        // last shard either samples a token or also fwd pass
        const gpt = gptShards[gptShards.len - 1];
        if (s >= inputs.len) {
            token = gpt.sample(s + 1, temp, token, state);
        } else {
            gpt.forward(s + 1, token, state);
        }
        const decoded_len = encoder.decode(&[_]usize{token}, state.decoded);
        std.debug.print("{s}", .{state.decoded[0..decoded_len]});
    }
}

pub fn main() !void {
    const temp = 0.8;
    // const hosts = [_]net.Address{ net.Address.resolveIp("0.0.0.0", 4005) catch unreachable, net.Address.resolveIp("0.0.0.0", 4006) catch unreachable };
    // const hosts = [_]net.Address{ net.Address.resolveIp("127.0.0.1", 4440) catch unreachable, net.Address.resolveIp("127.0.0.1", 4441) catch unreachable };
    // const hosts = [_]net.Address{ net.Address.resolveIp("10.42.0.1", 4440) catch unreachable, net.Address.resolveIp("10.42.0.1", 4441) catch unreachable };
    // const hosts = [_]net.Address{ net.Address.resolveIp("10.42.0.74", 4000) catch unreachable, net.Address.resolveIp("10.42.0.74", 4001) catch unreachable };

    var buf: [std.os.HOST_NAME_MAX]u8 = undefined;
    const hostname = try std.os.gethostname(&buf);
    var shardConfig: GPTShardConfig = undefined;
    const n1 = 2 * (modelConfig.n_layer / 3);
    if (std.mem.eql(u8, hostname, "ubuntu-machine")) {
        shardConfig = GPTShardConfig{
            .rank = 0,
            .serverPort = 5000,
            .nextAddress = try net.Address.resolveIp("10.42.0.2", 8001),
            .startBlock = 0,
            .endBlock = n1,
        };
    } else {
        shardConfig = GPTShardConfig{
            .rank = 1,
            .serverPort = 8001,
            .nextAddress = try net.Address.resolveIp("10.42.0.1", 5000),
            .startBlock = n1,
            .endBlock = modelConfig.n_layer,
        };
    }
    std.debug.print("Shard-{d} Port: {} Next: {} Layers: {d} to {d}\n", .{ shardConfig.rank, shardConfig.serverPort, shardConfig.nextAddress, shardConfig.startBlock, shardConfig.endBlock });

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();

    var inputs = try allocator.alloc(u32, modelConfig.context_size);
    var encoder = try load_encoder(allocator);
    defer encoder.deinit();
    const state = try State.init(modelConfig, allocator);
    // const gpts = &[_]GPTShard{ try GPTShard.load(config, shardConfigs[0], allocator), try GPTShard.load(config, shardConfigs[1], allocator) };
    const gpt = try GPTShard.load(modelConfig, shardConfig, allocator);

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    const prompt = args[1];

    const input_tokens = encoder.encode(prompt, inputs);
    const totalUsage = state.totalMemoryUsage();
    std.debug.print("Total memory usage of State instance: {d} bytes {d}mb\n", .{ totalUsage, totalUsage / (1024 * 1024) });

    try generateShardedTcpDistributed(
        gpt,
        encoder,
        temp,
        inputs[0..input_tokens],
        state,
    );
}

test "generateShardedTcpLocal" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    defer arena.deinit();
    const allocator = arena.allocator();
    var encoder = try load_encoder(allocator);
    defer encoder.deinit();
    const state = try State.init(modelConfig, allocator);
    var inputs = try allocator.alloc(usize, modelConfig.context_size);
    const testModelConfig = GPTConfig{ .name = @constCast("124M"), .vocab_size = 50257, .context_size = 32, .n_layer = 12, .n_heads = 12, .n_embed = 768 };
    const input_tokens = encoder.encode("Last weekend ", inputs);

    const shardConfigs = &[_]GPTShardConfig{
        GPTShardConfig{
            .rank = 0,
            .serverPort = 5000,
            .nextAddress = try net.Address.resolveIp("0.0.0.0", 8001),
            .startBlock = 0,
            .endBlock = modelConfig.n_layer / 2,
        },
        GPTShardConfig{
            .rank = 1,
            .serverPort = 8001,
            .nextAddress = try net.Address.resolveIp("0.0.0.0", 5000),
            .startBlock = modelConfig.n_layer / 2,
            .endBlock = modelConfig.n_layer,
        },
    };
    const gptShards = &[_]GPTShard{ try GPTShard.load(testModelConfig, shardConfigs[0], allocator), try GPTShard.load(testModelConfig, shardConfigs[1], allocator) };
    try generateShardedTcpLocal(gptShards, encoder, 0.5, inputs[0..input_tokens], state);
}

// pub fn main() !void {
//     // std.time.sleep(10 * std.time.ns_per_s);
//     const temp = 0.8;
//     // const modelConfig = GPTConfig.init(50257, 1024, 12, 12, 768);
//     // const config = GPTConfig.init(50257, 1024, 36, 20, 1280);
//     // const shardConfigs = [_]GPTShardConfig{ GPTShardConfig.init(0, 6), GPTShardConfig.init(6, 12) };
//     // const shardConfigs = [_]GPTShardConfig{ GPTShardConfig.init(0, config.n_layer / 2), GPTShardConfig.init(config.n_layer / 2, config.n_layer) };
//     const nShards = 2;
//     const hosts = [_]net.Address{ net.Address.resolveIp("0.0.0.0", 4005) catch unreachable, net.Address.resolveIp("0.0.0.0", 4006) catch unreachable };
//     // const hosts = [_]net.Address{ net.Address.resolveIp("127.0.0.1", 4440) catch unreachable, net.Address.resolveIp("127.0.0.1", 4441) catch unreachable };
//     // const hosts = [_]net.Address{ net.Address.resolveIp("10.42.0.1", 4440) catch unreachable, net.Address.resolveIp("10.42.0.1", 4441) catch unreachable };
//     // const hosts = [_]net.Address{ net.Address.resolveIp("10.42.0.74", 4000) catch unreachable, net.Address.resolveIp("10.42.0.74", 4001) catch unreachable };
//     const addresses = &blk: {
//         var arr: [nShards]net.Address = undefined;
//         var i: u16 = 0;
//         for (0..arr.len) |_| {
//             // arr[i] = try net.Address.resolveIp("10.42.0.1", 4000 + i);
//             arr[i] = hosts[i % hosts.len];
//             i += 1;
//         }
//         break :blk arr;
//     };
//     const shardConfigs = &blk: {
//         var arr: [nShards]GPTShardConfig = undefined;
//         const perLayer = modelConfig.n_layer / nShards;
//         var i: usize = 0;
//         var j: usize = 0;
//         while (j < modelConfig.n_layer and i < nShards) : (j += perLayer) {
//             const endLayer = if (i == nShards - 1) modelConfig.n_layer else j + perLayer;
//             arr[i] = GPTShardConfig{
//                 .rank = i,
//                 .address = addresses[i],
//                 .nextAddress = addresses[(i + 1) % nShards],
//                 .startBlock = j,
//                 .endBlock = endLayer,
//             };
//             std.debug.print("Shard {d} [{}] Layers {d} to {d}\n", .{ i, addresses[i], j, endLayer });
//             i += 1;
//         }
//         break :blk arr;
//     };

//     for (shardConfigs, 0..) |sc, i| {
//         std.debug.print("shard-{} : {} to {} : {} layers\n", .{ i, sc.startBlock, sc.endBlock, sc.endBlock - sc.startBlock });
//     }

//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     var inputs = try allocator.alloc(usize, modelConfig.context_size);
//     var encoder = try load_encoder(allocator);
//     defer encoder.deinit();
//     const state = try State.init(modelConfig, allocator);
//     // const gpts = &[_]GPTShard{ try GPTShard.load(config, shardConfigs[0], allocator), try GPTShard.load(config, shardConfigs[1], allocator) };
//     const gpts = &blk: {
//         var arr: [shardConfigs.len]GPTShard = undefined;
//         for (0..arr.len) |i| {
//             arr[i] = try GPTShard.load(
//                 modelConfig,
//                 shardConfigs[i],
//                 allocator,
//             );
//         }
//         break :blk arr;
//     };

//     const args = try std.process.argsAlloc(allocator);
//     defer std.process.argsFree(allocator, args);
//     const prompt = args[1];

//     const input_tokens = encoder.encode(prompt, inputs);
//     const totalUsage = state.totalMemoryUsage();
//     std.debug.print("Total memory usage of State instance: {d} bytes {d}mb\n", .{ totalUsage, totalUsage / (1024 * 1024) });

//     try generateShardedTcpLocal(
//         gpts,
//         encoder,
//         temp,
//         inputs[0..input_tokens],
//         state,
//     );
// }

// const GPT = struct {
//     const Self = @This();

//     config: GPTConfig,
//     wte: ops.Embedding,
//     wpe: ops.Embedding,
//     h: []const Block,
//     ln_f: ops.LayerNorm,
//     lm_head: ops.Linear,

//     pub fn init(
//         config: GPTConfig,
//         wte: ops.Embedding,
//         wpe: ops.Embedding,
//         h: []const Block,
//         ln_f: ops.LayerNorm,
//         lm_head: ops.Linear,
//     ) Self {
//         return Self{
//             .config = config,
//             .wte = wte,
//             .wpe = wpe,
//             .h = h,
//             .ln_f = ln_f,
//             .lm_head = lm_head,
//         };
//     }

//     /// Computes the forward pass and writes the result in state.logits.
//     pub fn forward(self: Self, seq_len: usize, token: usize, compute_logits: bool, state: State) void {
//         self.wpe.forward(&[1]usize{seq_len - 1}, state.pos_emb);
//         self.wte.forward(&[1]usize{token}, state.x);
//         for (0..self.config.n_embed) |i| {
//             state.x[i] += state.pos_emb[i];
//         }

//         // Forward the transformer.
//         for (0..self.h.len) |i| {
//             self.h[i].forward(seq_len, state.x, state);
//         }
//         self.ln_f.forward(state.x);

//         // Compute logits.
//         if (compute_logits) {
//             self.lm_head.forward(state.x, state.logits);
//         }
//     }

//     /// Samples the next token.
//     pub fn sample(self: Self, seq_len: usize, temp: f32, token: usize, state: State) usize {
//         self.forward(seq_len, token, true, state);
//         for (0..state.logits.len) |i| {
//             state.logits[i] /= temp;
//         }
//         ops.softmax(state.logits);
//         var rng = std.rand.DefaultPrng.init(@intCast(std.time.timestamp()));
//         var random = rng.random();
//         return random.weightedIndex(f32, state.logits);
//     }
// };
// pub fn load_gpt(config: GPTConfig, allocator: std.mem.Allocator) !GPT {
//     const wte = try load_embedding("wte", config.vocab_size, config.n_embed, allocator);
//     const wpe = try load_embedding("wpe", config.context_size, config.n_embed, allocator);
//     var h = try allocator.alloc(Block, config.n_layer);
//     for (0..h.len) |i| {
//         h[i] = try load_block(i, config, allocator);
//     }
//     const ln_f = try load_layer_norm("ln_f", config.n_embed, allocator);
//     const lm_head = ops.Linear.init(config.n_embed, config.vocab_size, wte.weight, null);
//     return GPT.init(config, wte, wpe, h, ln_f, lm_head);
// }
// pub fn main() !void {
//     const temp = 0.8;
//     // const config = GPTConfig.init(50257, 1024, 12, 12, 768);
//     // const config = GPTConfig.init(50257, 1024, 24, 16, 1024);
//     const config = GPTConfig.init(50257, 1024, 36, 20, 1280);
//     // const config = GPTConfig.init(50257, 1024, 48, 25, 1600);

//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     var inputs = try allocator.alloc(usize, config.context_size);
//     var encoder = try load_encoder(allocator);
//     defer encoder.deinit();
//     const state = try State.init(config, allocator);
//     const gpt = try load_gpt(config, allocator);

//     const args = try std.process.argsAlloc(allocator);
//     defer std.process.argsFree(allocator, args);
//     const prompt = args[1];

//     const input_tokens = encoder.encode(prompt, inputs);
//     generate(
//         gpt,
//         encoder,
//         temp,
//         inputs[0..input_tokens],
//         state,
//     );
// }
// pub fn main() !void {
//     const temp = 0.8;
//     const config = GPTConfig.init(50257, 1024, 12, 12, 768);
//     const shardConfig = GPTShardConfig.init(0, 12);

//     var gpa = std.heap.GeneralPurposeAllocator(.{}){};
//     var arena = std.heap.ArenaAllocator.init(gpa.allocator());
//     defer arena.deinit();
//     const allocator = arena.allocator();

//     var inputs = try allocator.alloc(usize, config.context_size);
//     var encoder = try load_encoder(allocator);
//     defer encoder.deinit();
//     const state = try State.init(config, allocator);
//     const gpt = try GPTShard.load(config, shardConfig, allocator);

//     const args = try std.process.argsAlloc(allocator);
//     defer std.process.argsFree(allocator, args);
//     const prompt = args[1];

//     const input_tokens = encoder.encode(prompt, inputs);
//     generate(
//         gpt,
//         encoder,
//         temp,
//         inputs[0..input_tokens],
//         state,
//     );
// }
