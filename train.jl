include("data.jl")
include("rwkv.jl")
include("utils.jl")

using Flux.Losses
using OneHotArrays
using Wandb, Dates, Logging
using CUDA
using BSON: @save
using Optimisers

function save_model(model, path)
    model = model |> cpu
    @save path model
end

cfg = (
    learning_rate = 3e-4,
    epochs = 5,
    dataset = "astroph_combined.jsonl",
    use_cuda = true,
    ctx_len = 256,
    batch_size = 8,
    log_per_nbatch = 5,
    save_per_nbatch = 1000,
    n_vocab = 50277,
    n_layer = 12,
    n_head = 12
)
lg = WandbLogger(
    project = "nano-llm.jl",
    name = "rwkv-169m-$(now())",
    config = Dict(
        "learning_rate" => cfg.learning_rate,
        "dataset" => "astroph_combined.jsonl",
        "n_layer" => cfg.n_layer,
        "n_head" => cfg.n_head,
    ),
)
global_logger(lg)

data_file = cfg.dataset
tokenizer = get_tokenizer()
ts = TextSplitter(cfg.ctx_len+1, 16, tokenizer)

function loss_func(y_pred, y; n_vocab=cfg.n_vocab)
    return logitcrossentropy(y_pred, onehotbatch(y, 1:n_vocab))
end

if cfg.use_cuda && CUDA.functional()
    CUDA.allowscalar(false)
    @info "Using CUDA"
    device = gpu
else
    @info "Using CPU"
    device = identity
end

Flux.trainable(m::RWKV) = (blocks=m.blocks,)
Flux.trainable(m::Block) = (token_mixing=m.token_mixing,)

# device = identity  # debug
model = rwkv_from_pth("RWKV-4-Pile-169M-20220807-8023.pth"; cfg.n_layer) |> f32
model = model |> device

# setup optimizer
opt = Optimisers.setup(Optimisers.Adam(cfg.learning_rate), model)
i_batch = 1
for epoch in 1:cfg.epochs
    global i_batch
    batches = jsonl_reader(data_file) |> ch->batch_sampler(ch, ts; batch_size=cfg.batch_size)
    for (x, y) in batches
        x = x |> device
        y = y |> device
        val, grads = Flux.withgradient(model) do m
            y_pred = m(x)
            loss_func(y_pred, y)
        end
        Flux.update!(opt, model, grads[1])
        if i_batch % cfg.log_per_nbatch == 0
            @info "metrics" loss=val
            println("Ep $(epoch) Batch $(i_batch) Loss $(val)")
        end
        if i_batch % cfg.save_per_nbatch == 0
            @info "saving model"
            save_model(model, "rwkv-169m-$(now()).bson")
        end
        i_batch += 1
    end
end

close(lg)