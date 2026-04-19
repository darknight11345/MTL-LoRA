import argparse
import ast
import copy
import json
import os
import re
import sys
from scipy.stats import pearsonr
import torch
from src.custom_model import LlamaForCausalLM
from src.utils import wrap_model
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaTokenizer,
)

#sys.path.append(os.path.join(os.getcwd(), "~/MTL-LoRA"))

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


task_name_to_id = {
    'econometrics': 0, 'high_school_macroeconomics': 1, 'high_school_microeconomics': 2, 'business_ethics': 3, 'management': 4, 'marketing': 5, 'abstract_algebra': 6, 'college_mathematics': 7, 'elementary_mathematics': 8, 'high_school_mathematics': 9, 'high_school_statistics': 10
}


def main():
    args = parse_args()

    def evaluate(
        instructions,
        input=None,
        temperature=1.0,
        top_p=1,
        top_k=1,
        num_beams=1,
        max_new_tokens=32,
        **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)

        if args.adapter in ["mlora", "moelora"]:
            lambda_index = task_name_to_id[args.dataset]
            lambda_index = (
                torch.tensor(lambda_index).repeat(input_ids.shape[0]).to(device)
            )
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=False,
            **kwargs,
        )
        with torch.no_grad():
            if args.adapter in ["mlora", "moelora"]:
                #print(f"Model Type: {type(model)}")
                #print(f"Available Methods: {dir(model)}")

                generation_output = model.generate(
                    input_ids=input_ids,
                    lambda_index=lambda_index,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                )
            else:
                generation_output = model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_tokens,
                    use_cache=False,
                )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True) # the list of decoded string based on  the token s
        #print("tha ouputs before split in evaluate is:",outputs)
        outputs = [o.split("### Response:")[1].strip() for o in outputs] #this gives answer like A or B ...
        #print("tha ouputs after split in evaluate is:",outputs) 
        return outputs

    save_file = f"experiment/{args.model}-{args.adapter}-{args.dataset}.json"
    os.makedirs("experiment", exist_ok=True)

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    tokenizer, model = load_model(args)

    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    gold_scores = []
    pred_scores = []
    class_labels = []
    class_preds = []
    
    task_predictions = {}
    #pbar = tqdm(total=total)
    model.to(torch.bfloat16)
    model.eval()
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get("instruction") for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            task_id = data["task_id"]
            label = data.get("output") 
            flag = False
            predict = extract_answer(args, output)
            
            # classification
            flag = (label == predict)
            correct += int(flag)
            class_labels.append(label)
            class_preds.append(predict)
            
            # per-task predictions
            if task_id not in task_predictions:
                task_predictions[task_id] = ([], [])
            task_predictions[task_id][0].append(label)    # true
            task_predictions[task_id][1].append(predict)  # predicted

                    
            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = predict
            new_data["flag"] = flag
            output_data.append(new_data)
            #print(data["instruction"])
            #print(output)
            #print("prediction:", predict)
            #print("label:", label)
        #print("---------------")
        with open(save_file, "w+") as f:
            json.dump(output_data, f, indent=4)
        #pbar.update(1)
        
    if args.dataset == "stsb":
        pearson, _ = pearsonr(pred_scores, gold_scores)
        print("STS-B Pearson:", pearson)
        mae = np.mean(np.abs(np.array(pred_scores) - np.array(gold_scores)))
        rmse = np.sqrt(np.mean((np.array(pred_scores) - np.array(gold_scores))**2))
        print(f"STS-B Metrics → Pearson: {pearson:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    else:
        #accuracy = correct / current
        #print("Overall Accuracy:", accuracy)
        print("Unique true labels:", set(class_labels))
        print("Unique predicted labels:", set(class_preds))
        # pair gold and predicted
        filtered_pairs = [(g, p) for g, p in zip(class_labels, class_preds) if p is not None]
        if not filtered_pairs:
            raise ValueError("All predictions failed parsing; nothing to evaluate.")

        filtered_labels, filtered_preds = zip(*filtered_pairs)
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(filtered_labels, filtered_preds, average='macro',zero_division=0)
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        # Confusion matrix
        labels_sorted = sorted(list(set(filtered_labels)))  # ensures consistent order
        cm = confusion_matrix(filtered_labels, filtered_preds, labels=labels_sorted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix for {args.dataset}")
        #plt.show()
        
        # save to file
        conf_matrix_file = f"{save_file.replace('.json','')}_confusion_matrix.png"
        plt.savefig(conf_matrix_file)
        plt.close()  # close the figure to free memory
        
        # Per-task accuracy
        print("\nPer-task Accuracy:")
        for task_id, (true_labels, preds) in task_predictions.items():
            task_acc = accuracy_score(true_labels, preds)
            print(f"Task {task_id}: Accuracy = {task_acc:.4f}")
        
    #pbar.close()
    print("\n")
    print("test finished")


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
Example 1:
Question: If the p-value of a coefficient is 0.03, what does it imply at 5% significance?
Options:
A. Fail to reject H0
B. Reject H0
C. Coefficient is zero
D. Coefficient is irrelevant
Answer: B
Now answer this question:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:
                """  # noqa: E501


def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    
    file_path = f"/pfs/data6/home/ul/ul_student/ul_swv79/MTLLoRA/Code/MTL-LoRA/dataset/{args.dataset}.json"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, "r"))
    return json_data


def create_batch(dataset, batch_size):
    batches = []
    num_batch = (
        len(dataset) // batch_size
        if len(dataset) % batch_size == 0
        else len(dataset) // batch_size + 1
    )
    for i in range(num_batch):
        batch = dataset[i * batch_size : min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[
            "elementary_mathematics", "business_ethics", "econometrics", "abstract_algebra", "college_mathematics", "marketing", "management", "high_school_mathematics", "high_school_statistics", "high_school_macroeconomics", "high_school_microeconomics"
       ],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=["LLaMA-7B", "LLaMA-13B", "BLOOM-7B", "GPT-j-6B"],
        required=True,
    )
    parser.add_argument(
        "--adapter",
        choices=["mlora", "moelora", "multilora", "dora"],
        required=True,
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_weights", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lora_target_modules", type=ast.literal_eval, required=True)
    parser.add_argument("--lora_r", type=int, required=True)
    parser.add_argument("--lora_alpha", type=int, required=True)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # mlora
    parser.add_argument("--lambda_num", type=int)
    parser.add_argument("--num_B", type=int)
    parser.add_argument("--temperature", type=float)
    # multilora
    parser.add_argument("--lora_num", type=int)
    # moelora
    parser.add_argument("--expert_num", type=int)
    parser.add_argument("--task_num", type=int)
    parser.add_argument("--te_dim", type=int)
    # dora hyperparams
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=False,
        help="merge weights",
    )
    parser.add_argument(
        "--Wdecompose",
        type=bool,
        default=False,
        help="Wdecompose",
    )
    parser.add_argument(
        "--dora_simple",
        type=bool,
        default=True,
        help="dora simple",
    )

    return parser.parse_args()


def load_model(args) -> tuple:
    """
    load tuned model
    Args:
        args:

    Returns:
        tuple(tokenizer, model)
    """
    base_model = args.base_model
    if not base_model:
        raise ValueError(f"can not find base model name by the value: {args.model}")
    lora_weights = args.lora_weights
    if not lora_weights:
        raise ValueError(f"can not find lora weight, the value is: {lora_weights}")

    if "LLaMA" in args.model:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if "llama" in base_model.lower() and args.adapter.lower() in [
        "mlora",
        "moelora",
    ]:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )
        #print(type(model))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
            trust_remote_code=True,
        )  # fix zwq

    if args.adapter.lower() == "mlora":
        mlora_config = {
            "type": "mlora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lambda_num": args.lambda_num,
            "B_num": args.num_B,
            "B_scale": args.temperature,
            "diagonal_format": False,
        }
    elif args.adapter.lower() == "multilora":
        mlora_config = {
            "type": "multilora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_num": args.lora_num,
        }
    elif args.adapter.lower() == "moelora":
        mlora_config = {
            "type": "moelora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "expert_num": args.expert_num,
            "task_num": args.task_num,
            "task_embedding_dim": args.te_dim,
        }
    elif args.adapter.lower() == "dora":
        mlora_config = {
            "type": "dora",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "merge_weights": args.merge_weights,
            "Wdecompose": args.Wdecompose,
            "dora_simple": args.dora_simple,
        }

    model = wrap_model(model, args.lora_target_modules, mlora_config)

    state_dict = torch.load(lora_weights, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg.unexpected_keys)
    assert len(msg.unexpected_keys) == 0

    for name, param in model.named_parameters():
        param.requires_grad = False

    return tokenizer, model


def load_instruction(args) -> str:
    instruction = ""
    if not instruction:
        raise ValueError("instruct not initialized")
    return instruction


def extract_answer(args, sentence: str) -> str| None:
    #dataset = args.dataset
    #if dataset == "sst2":
    #sentence_ = sentence.lower().strip()
    #print(" the predicted sentence is:", sentence)
    if args.dataset == "business_ethics":
        print(" the predicted sentence is:", sentence)
        pred_answers = re.findall(r"A|B|C", sentence)
    else:
        pred_answers = re.findall(r"A|B|C|D", sentence)

    if not pred_answers:
        return None
    else:
        return pred_answers[0]
    


if __name__ == "__main__":
    main()
