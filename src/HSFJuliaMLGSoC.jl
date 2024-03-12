module HSFJuliaMLGSoC

using CSV, Flux, JLD2, ProgressMeter, Statistics


"""
Parse the CSV file containing x,y,z and class fields

# Arguments
- `csvpath::String`: Path to CSV file.
"""
parse_csv(csvpath::String) = CSV.File(csvpath; types=[Float32, Float32, Float32, Bool], truestrings=["s"], falsestrings=["b"])


"""
Convert dataset from parsed CSV to Flux DataLoader.

# Arguments
- `dataset::CSV.File`: Parsed dataset CSV file.
- `split_train_val::Bool=false`: Flag to split dataset into train and validation splits (85/15).
"""
function to_dataloader(dataset::CSV.File, split_train_val::Bool=false)
    # Separate predictors from labels
    coords = [dataset.x'; dataset.y'; dataset.z']
    labels = dataset.class'

    # Create dataloader (85/15 train/val if split flag is true)
    if split_train_val
        train_dataloader = Flux.DataLoader((coords[:,begin:85_000], labels[:,begin:85_000]), batchsize=64, shuffle=true)
        val_dataloader = Flux.DataLoader((coords[:,85_001:end], labels[:,85_001:end]), batchsize=64, shuffle=false)
        return train_dataloader, val_dataloader
    else
        return Flux.DataLoader((coords, labels), batchsize=64, shuffle=false)
    end
end


"""
Chosen MLP architecure from multiple trainings on notebook.
"""
model() = Chain(
    Dense(3 => 32, relu),
    Dense(32 => 16, relu),
    Dense(16 => 1, sigmoid)
)


"""
Calculate accuracy ``\\frac{TP+TN}{TP+TN+FP+FN}``.

Predictions are between 0 and 1, values above threshold of 0.5 are considered **signal**, otherwise are considered **background**.
Iterate through dataloader to compare predictions with ground-truths, then use reduce with hcat to combine all batches into a single BitMatrix,
where 1 is a true prediction (either TP or TN) and 0 is a false prediction (FP or FN). Given this, the accuracy is calculated with mean.

# Arguments
- `model`: Model to evaluate.
- `dataloader::Flux.MLUtils.DataLoader`: Dataset to be evaluated.
"""
accuracy(model, dataloader::Flux.MLUtils.DataLoader) = reduce(hcat, [(model(x) .> 0.5) .== y for (x,y) in dataloader]) |> mean


"""
Train binary classification model with progress meter and checkpoints.

# Arguments
- `model`: Model to train.
- `train_dataloader::Flux.MLUtils.DataLoader`: Dataset used for training.
- `val_dataloader::Flux.MLUtils.DataLoader`: Dataset used for validation after each epoch.
- `epochs::Integer=200`: Number of epochs.
- `optimiser::Flux.Optimise.AbstractOptimiser=Adam(0.01)`: Dataset to be evaluated.
"""
function train!(
    model,
    train_dataloader::Flux.MLUtils.DataLoader,
    val_dataloader::Flux.MLUtils.DataLoader,
    epochs::Integer=200,
    optimiser::Flux.Optimise.AbstractOptimiser=Adam(0.01)
)
    p = Progress(epochs*length(train_dataloader); showspeed=true)

    optim = Flux.setup(optimiser, model)
    val_acc = NaN

    for epoch in 1:epochs
        losses = []
        for (step, (x, y)) in enumerate(train_dataloader)
            loss, grads = Flux.withgradient(model) do m
                Flux.binarycrossentropy(m(x), y)
            end
            Flux.update!(optim, model, grads[begin])

            push!(losses, loss)
            next!(p, showvalues=[(:epoch, epoch), (:step,step), (:loss,mean(losses)), (:val_acc, val_acc)])
        end

        # Calculate accuracy over validation set
        val_acc = accuracy(model, val_dataloader)
    
        # Save model checkpoints
        if epoch % 50 == 0
            jldsave("model-checkpoint-$(epoch).jld2", model_state = Flux.state(model))
        end
    end
end

end # module HSFJuliaMLGSoC