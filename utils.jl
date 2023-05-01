using PyCall
using Flux
using StatsBase

include("rwkv.jl")
torch = pyimport("torch")

function rwkv_from_pth(pth_path="RWKV-4-Pile-169M-20220807-8023.pth"; n_layer=12)
    data = torch.load(pth_path, "cpu")

    ln_init = LN(
        data["blocks.0.ln0.weight"].float().numpy(),
        data["blocks.0.ln0.bias"].float().numpy(),
    )
    ln_final = LN(
        data["ln_out.weight"].float().numpy(),
        data["ln_out.bias"].float().numpy()
    )
    embedding = Embedding(
        data["emb.weight"].float().numpy()'
    )
    lm_head = Embedding(
        data["head.weight"].float().numpy()'
    )
    blocks = []
    for i = 0:n_layer-1
        ln1 = LN(
            data["blocks.$i.ln1.weight"].float().numpy(),
            data["blocks.$i.ln1.bias"].float().numpy(),
        )
        ln2 = LN(
            data["blocks.$i.ln2.weight"].float().numpy(),
            data["blocks.$i.ln2.bias"].float().numpy(),
        )
        time_first = data["blocks.$i.att.time_first"].float().numpy()
        time_decay = -exp.(data["blocks.$i.att.time_decay"].float().numpy())
        token_mixing = TokenMixing(
            dropdims(data["blocks.$i.att.time_mix_k"].float().numpy(), dims=(1,2)),
            dropdims(data["blocks.$i.att.time_mix_v"].float().numpy(), dims=(1,2)),
            dropdims(data["blocks.$i.att.time_mix_r"].float().numpy(), dims=(1,2)),
            Dense(
                data["blocks.$i.att.receptance.weight"].float().numpy(),
                false
            ), # r_proj
            Dense(
                data["blocks.$i.att.key.weight"].float().numpy(),
                false
            ), # k_proj
            Dense(
                data["blocks.$i.att.value.weight"].float().numpy(),
                false
            ), # v_proj
            Dense(
                data["blocks.$i.att.output.weight"].float().numpy(),
                false
            ), # out_proj
            time_first,
            time_decay,
        )
        channel_mixing = ChannelMixing(
            dropdims(data["blocks.$i.ffn.time_mix_k"].float().numpy(), dims=(1,2)),
            dropdims(data["blocks.$i.ffn.time_mix_r"].float().numpy(), dims=(1,2)),
            Dense(
                data["blocks.$i.ffn.receptance.weight"].float().numpy(),
                false
            ), # r_proj
            Dense(
                data["blocks.$i.ffn.key.weight"].float().numpy(),
                false
            ), # k_proj
            Dense(
                data["blocks.$i.ffn.value.weight"].float().numpy(),
                false
            ), # v_proj
        )
        push!(blocks, Block(
            ln1,
            token_mixing,
            ln2,
            channel_mixing
        ))
    end
    RWKV(
        ln_init,
        embedding,
        blocks,
        ln_final,
        lm_head
    )
end

# hacky
function get_tokenizer()
    py"""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file('20B_tokenizer.json')
    """
    py"tokenizer"
end

function sample_logits(logits; temperature=1.0, top_p=0.9, use_argmax=false)
    if use_argmax
        return argmax(logits)
    end
    probs = softmax(logits; dims=1)
    sorted_probs = sort(probs; rev=true)
    cum_probs = cumsum(sorted_probs)
    cutoff = sorted_probs[argmax(cum_probs .> top_p)]
    probs = (probs .> cutoff) .* probs
    if temperature != 1.0
        probs .^= 1/temperature
    end
    probs = probs / sum(probs)

    sample(collect(1:length(probs)), ProbabilityWeights(probs)) |> Int
end