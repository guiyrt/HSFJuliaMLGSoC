import HSFJuliaMLGSoC, Flux.loadmodel!, JLD2.load
using ArgParse

"""
Argument parser for model training/evaluation.
"""
function parse_commandline()
    s = ArgParseSettings(
        prog="Train/evaluation script for hsf-julia-ml-gsoc evaluation task.",
        description="Dataset CSV path must always be passed as first argument, but model training can be skipped if a JLD2 weights file "*
                    "is provided with -w option. After model training OR loading, the accuracy over the whole dataset is printed."
    )

    @add_arg_table s begin
        "dataset"
            help = "Path to CSV file representing dataset"
            arg_type = String
            required = true
        "-e", "--epochs"
            help = "Number of epoch to train the model"
            arg_type = Int
            default = 200
        "-w", "--modelweights"
            help = "Path to JLD2 file containing pre-trained weights."
            arg_type = String
    end

    return parse_args(s)
end


# Parse arguments
args = parse_commandline()

# Read dataset
dataset = HSFJuliaMLGSoC.parse_csv(args["dataset"])

# Create dataloaders
dataloader = HSFJuliaMLGSoC.to_dataloader(dataset)

# Create MLP model
model = HSFJuliaMLGSoC.model()

# Train model if no weights are provided, load them otherwise
if args["modelweights"] |> isnothing
    # Data split for training and validation after each epoch
    train_dataloader, val_dataloader = HSFJuliaMLGSoC.to_dataloader(dataset, true)
    HSFJuliaMLGSoC.train!(model, train_dataloader, val_dataloader, args["epochs"])
else
    loadmodel!(model, load(args["modelweights"], "model_state"))
end

HSFJuliaMLGSoC.accuracy(model, dataloader) |> println