import argparse
import os

import torch
import mteb
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# add args:
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Banking77Classification")
parser.add_argument("--model_name", type=str, default="BAAI/bge-small-en-v1.5")
parser.add_argument("--config", type=str, default="mean", choices=["replace", "concat", "mean", "no"])

args = parser.parse_args()

config = args.config

original_model = SentenceTransformer(args.model_name)
linear_layer = nn.Linear(original_model.get_sentence_embedding_dimension(),
                         original_model.get_sentence_embedding_dimension(), bias=False)
nn.init.normal_(linear_layer.weight)
for param in linear_layer.parameters():
    param.requires_grad = False


class CustomModel(type(original_model)):  # Inherit directly from the original model's class
    def __init__(self, original_model):
        super().__init__()
        self.__dict__ = original_model.__dict__.copy()  # Copy all attributes from the original model
        linear_layer.to(self.device)

    def encode(self, x, *args, **kwargs):
        x = super().encode(x, *args, **kwargs)
        x = torch.tensor(x).to(self.device)
        if config == "replace":
            x = linear_layer(x)
        elif config == "concat":
            x = torch.cat((x, linear_layer(x)), dim=1)
        elif config == "mean":
            x = (x + linear_layer(x)) / 2
        elif config == "no":
            pass
        else:
            raise ValueError(f"Unknown config: {config}")
        x = x.cpu().detach().numpy()
        return x


model = CustomModel(original_model)
#
# def apply_linear_layer_hook(module, input, output):
#     linear_layer.to(output['sentence_embedding'].device)
#     if args.config == "replace":
#         output['sentence_embedding'] = linear_layer(output['sentence_embedding'])
#         output['token_embeddings'] = linear_layer(output['token_embeddings'])
#     elif args.config == "concat":
#         output['sentence_embedding'] = torch.cat(
#             (output['sentence_embedding'], linear_layer(output['sentence_embedding'])), dim=1)
#         output['token_embeddings'] = torch.cat((output['token_embeddings'], linear_layer(output['token_embeddings'])),
#                                                dim=1)
#     elif args.config == "mean":
#         output['sentence_embedding'] = (output['sentence_embedding'] + linear_layer(output['sentence_embedding'])) / 2
#         output['token_embeddings'] = (output['token_embeddings'] + linear_layer(output['token_embeddings'])) / 2
#     elif args.config == "no":
#         pass
#     return output
# hook_handle = original_model[2].register_forward_hook(apply_linear_layer_hook)
evaluation = mteb.MTEB(tasks=mteb.get_tasks(tasks=[args.task]))
os.makedirs(f"./results/{args.task}_{args.config}/", exist_ok=True)
output_folder = f"./results/{args.task}_{args.config}/"
output_csv = f"./results/all.csv"
results = evaluation.run(model, output_folder=output_folder, overwrite_results=True)
for res in results:
    task = res.task_name
    score = res.scores['test'][0]['main_score']
    with open(output_csv, "a") as f:
        f.write(f"{task},{args.model_name},{args.config},{score}\n")
