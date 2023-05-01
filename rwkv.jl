using Flux
using MLUtils: batch, zeros_like
using NNlibCUDA: batched_mul, softmax, sigmoid, pad_zeros
using Statistics

struct LN
    γ
    β
    ϵ
    dims
end
LN(γ, β) = LN(γ, β, 1e-5, 1)
LN(n::Integer) = LN(ones(Float32, n), zeros(Float32, n))
@Flux.functor LN
Flux.trainable(m::LN) = (γ=m.γ, β=m.β)
function (m::LN)(x::AbstractArray{T,N}) where {T,N}
    μ = mean(x, dims=m.dims)
    σ² = var(x, dims=m.dims)
    (T).(m.γ .* (x .- μ) ./ (σ² .+ m.ϵ).^0.5 .+ m.β)
end

time_mix(x, x_prev, mix) = @. x * mix + x_prev * (1 - mix)
square_relu(x::T) where T = max(zero(T), x)^2
function exp_mix(v1::AbstractArray{T}, v2::AbstractArray{T}, p1::AbstractArray{T}, p2::AbstractArray{T}) where T
    p = max.(p1, p2)
    (@. exp(p1 - p) * v1 + exp(p2 - p) * v2, p)
end

# Collectively rescale `v1` and `v2` to better dynamic range
function exp_selfadj(v1, v2, p)
    p_new = @. max(min(abs(asinh(v1)), p), min(abs(asinh(v2)), p))
    factor = @. exp(p - p_new)
    @. (v1*factor, v2*factor, p_new)
end

mutable struct State
    x_tm  # token mixing
    x_cm  # channel mixing
    a # numerator
    b # denominator
    p # largest exponent seen
end

State(n_embed::Integer, n_layer::Integer) = begin
    dim = (n_embed, n_layer)
    State(zeros(Float32, dim), zeros(Float32, dim), zeros(Float32, dim), zeros(Float32, dim), zeros(Float32, dim))
end
@Flux.functor State

function recur_step(left::Vector, right::Vector; w)
    a_prev, b_prev, p_prev = left
    expkv, expk, p = right
    a_new, p_new = exp_mix(a_prev, expkv, p_prev .+ w, p)
    b_new, _ = exp_mix(b_prev, expk, p_prev .+ w, p)
    [a_new, b_new, p_new]
end

struct TokenMixing{T}
    Tₖ::AbstractArray{T, 1}
    Tᵥ::AbstractArray{T, 1}
    Tᵣ::AbstractArray{T, 1}
    r_proj
    k_proj
    v_proj
    out_proj 
    time_first::AbstractArray{T, 1}
    time_decay::AbstractArray{T, 1}  # <-- w
end

@Flux.functor TokenMixing

TokenMixing(n_embed::Integer) = TokenMixing(
    zeros(Float32, n_embed), # Tₖ
    zeros(Float32, n_embed), # Tᵥ
    zeros(Float32, n_embed), # Tᵣ
    Dense(n_embed, n_embed, bias=false), # r_proj
    Dense(n_embed, n_embed, bias=false), # k_proj
    Dense(n_embed, n_embed, bias=false), # v_proj
    Dense(n_embed, n_embed, bias=false), # out_proj
    zeros(Float32, n_embed), # time first
    ones(Float32, n_embed),  # time_decay
)

function (m::TokenMixing)(x::AbstractArray{T,2}, state::State; i) where T
    n_embed, n_seq = size(x)

    x_prev = hcat(@views(state.x_tm[:, i]), @views(x[:, 1:end-1]))

    xₖ = time_mix(x, x_prev, m.Tₖ)
    xᵥ = time_mix(x, x_prev, m.Tᵥ)
    xᵣ = time_mix(x, x_prev, m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ)
    v = m.v_proj(xᵥ)

    p = k
    expk = k.*0 .+ 1
    expkv = v

    step_f = (a, b) -> recur_step(a, b; w=m.time_decay)
    a_prev, b_prev, p_prev = @views(state.a[:, i]), @views(state.b[:, i]), @views(state.p[:, i])
    substrate = [[@views(expkv[:,i]), @views(expk[:,i]), @views(p[:,i])] for i = 1:n_seq]
    abp = accumulate(step_f, substrate; init=[a_prev, b_prev, p_prev])

    a_prev = batch([a_prev, [abp[i][1] for i = 1:n_seq-1]...])
    b_prev = batch([b_prev, [abp[i][2] for i = 1:n_seq-1]...])
    p_prev = batch([p_prev, [abp[i][3] for i = 1:n_seq-1]...])

    c, _ = exp_mix(a_prev, expkv, p_prev, p .+ m.time_first)
    d, _ = exp_mix(b_prev, expk,  p_prev, p .+ m.time_first)
    rwkv = @. r * c / d

    # update state
    @views state.x_tm[:, i] .= x[:, end]
    @views state.a[:, i] .= abp[end][1]
    @views state.b[:, i] .= abp[end][2]
    @views state.p[:, i] .= abp[end][3]

    m.out_proj(rwkv), state
end
    
# stateless for training
function (m::TokenMixing)(x::AbstractArray{T,3}) where T
    n_embed, n_batch, n_seq = size(x)

    @views x_prev = pad_zeros(x, (0,0,0,0,1,0))[:, :, 1:end-1]

    xₖ = time_mix(x, x_prev, m.Tₖ)
    xᵥ = time_mix(x, x_prev, m.Tᵥ)
    xᵣ = time_mix(x, x_prev, m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ)
    v = m.v_proj(xᵥ)

    p = k
    expk = zeros_like(k) .+ 1
    expkv = v

    step_f = (a, b) -> recur_step(a, b; w=m.time_decay)
    a_prev = b_prev = p_prev = zeros_like(k, eltype(v), (n_embed, n_batch))
    substrate = @views [[expkv[:,:,i], expk[:,:,i], p[:,:,i]] for i = 1:n_seq]
    abp = accumulate(step_f, substrate; init=[a_prev, b_prev, p_prev])

    a_prev = batch([a_prev, [abp[i][1] for i = 1:n_seq-1]...])
    b_prev = batch([b_prev, [abp[i][2] for i = 1:n_seq-1]...])
    p_prev = batch([p_prev, [abp[i][3] for i = 1:n_seq-1]...])

    c, _ = exp_mix(a_prev, expkv, p_prev, p .+ m.time_first)
    d, _ = exp_mix(b_prev, expk,  p_prev, p .+ m.time_first)
    rwkv = @. r * c / d

    m.out_proj(rwkv)
end

struct ChannelMixing{T}
    Tₖ::AbstractArray{T, 1}  # will be taken out in the future
    Tᵣ::AbstractArray{T, 1}  # will be taken out in the future
    r_proj
    k_proj
    v_proj
end

@Flux.functor ChannelMixing

ChannelMixing(n_embed::Integer) = ChannelMixing(
    zeros(Float32, n_embed), # Tₖ
    zeros(Float32, n_embed), # Tᵣ
    Dense(n_embed, n_embed, bias=false), # r_proj
    Dense(n_embed, n_embed*4, bias=false), # k_proj
    Dense(n_embed*4, n_embed, bias=false), # v_proj
)

function (m::ChannelMixing)(x::AbstractArray{T, 2}, state::State; i) where T
    n_embed, n_seq = size(x)

    x_prev = @views(state.x_cm[:, i])
    if size(x, 2) > 1
        x_prev = hcat(x_prev, @views(x[:, 1:end-1]))
    end
    xₖ = time_mix(x, x_prev, m.Tₖ)
    xᵣ = time_mix(x, x_prev, m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ) .|> square_relu

    # update state
    @views state.x_cm[:, i] .= x[:, end]

    r .* (m.v_proj(k)), state
end

# stateless for training purpose
function (m::ChannelMixing)(x::AbstractArray{T, 3}) where T
    # n_embed, n_batch, n_seq = size(x)
    @views x_prev = pad_zeros(x, (0,0,0,0,1,0))[:, :, 1:end-1]

    xₖ = time_mix(x, x_prev, m.Tₖ)
    xᵣ = time_mix(x, x_prev, m.Tᵣ)

    r = m.r_proj(xᵣ) .|> sigmoid
    k = m.k_proj(xₖ) .|> square_relu

    r .* (m.v_proj(k))
end

struct Block
    ln1
    token_mixing
    ln2
    channel_mixing
end

@Flux.functor Block

Block(n_embed::Integer) = Block(
    LN(n_embed),
    TokenMixing(n_embed),
    LN(n_embed),
    ChannelMixing(n_embed),
)

function (m::Block)(x, state::State; i)
    xp, state = m.token_mixing(m.ln1(x), state; i=i)
    x = x + xp
    xp, state = m.channel_mixing(m.ln2(x), state; i=i)
    x = x + xp 
    x, state
end

function (m::Block)(x)
    xp = m.token_mixing(m.ln1(x))
    x = x + xp
    xp = m.channel_mixing(m.ln2(x))
    x = x + xp
    x
end

struct RWKV
    ln_init
    embedding
    blocks
    ln_final
    lm_head
end

@Flux.functor RWKV

RWKV(n_embed::Integer, n_blocks::Integer, n_vocab::Integer) = RWKV(
    Embedding(n_vocab, n_embed),
    LN(n_embed),
    [Block(n_embed) for _ in 1:n_blocks],
    LN(n_embed),
    Embedding(n_embed, n_vocab)
)

(m::RWKV)(x::AbstractArray{T, 1}, state::State) where T = begin
    x = m.embedding(x)
    x = m.ln_init(x)
    for i in 1:length(m.blocks)
        x, state = m.blocks[i](x, state; i=i)
    end
    x = m.ln_final(x)

    # x: [n_embed, n_seq]
    x = m.lm_head.weight' * x

    x, state
end
(m::RWKV)(x::Integer, state::State) = begin
    out, state = m([x], state)
    out[:, end], state
end

# stateless for training purpose
(m::RWKV)(x::AbstractArray{T, 2}) where T = begin
    x = m.embedding(x)
    # better performance
    x = permutedims(x, (1, 3, 2))
    x = m.ln_init(x)
    for i in 1:length(m.blocks)
        x = m.blocks[i](x)
    end
    x = m.ln_final(x)
    # better performance
    x = permutedims(x, (1, 3, 2))
    # x: [n_embed, n_seq, n_batch]
    batched_mul(m.lm_head.weight', x)
end
