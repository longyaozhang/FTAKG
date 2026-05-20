import argparse
import os
import yaml

from schedules import S2L  # 仅导入 S2L
from utils import get_tokenizer, smart_tokenizer_and_embedding_resize, get_model, rank0_print


def set_default_values(args):
    # 设置默认值 (保留原 train.py 逻辑)
    if "ref_model_path" not in args:
        args["ref_model_path"] = None
    if "n_components" not in args:
        args["n_components"] = -1
    if "num_loss_ckpts" not in args:
        args["num_loss_ckpts"] = -1
    if "distance" not in args:
        args["distance"] = 'euclidean'
    if "seed" not in args:
        args["seed"] = 42

    return args


def main(config_file):
    # 加载配置
    with open(config_file, 'r') as f:
        args = yaml.full_load(f)
    rank0_print('Configuration loaded!')

    # 强制设置或确认 schedule_name 为 S2L (虽然下面直接使用了S2L类，但为了保持args一致性)
    args["schedule_name"] = "S2L"

    # 设置默认参数
    args = set_default_values(args)
    rank0_print(yaml.dump(args, sort_keys=False, default_flow_style=False))

    # 创建必要的目录
    args["data_path_root"] = f"{args['result_dir_name']}/data"
    args["output_dir_root"] = f"{args['result_dir_name']}/output"
    os.makedirs(args["data_path_root"], exist_ok=True)
    os.makedirs(args["output_dir_root"], exist_ok=True)

    # 初始化 Tokenizer
    tokenizer, special_tokens_dict = get_tokenizer(
        model_name_or_path=args["model_name_or_path"],
        cache_dir=args["cache_dir"],
        model_max_length=args["model_max_length"]
    )
    rank0_print('*** Tokenizer initialized!')

    tokenizer, _ = smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=None
    )
    rank0_print('*** Smart tokenizer and embedding resize done!')

    # 初始化 S2L Schedule
    # 直接实例化 S2L，不再通过 get_schedule 选择
    schedule = S2L(
        model=None,
        tokenizer=tokenizer,
        args=args
    )
    rank0_print('*** S2L Schedule built!')

    # 初始化数据
    schedule.initialize_labeled_data()

    # 执行筛选并保存数据
    schedule.save_labeled_unlabeled_data()

    # 打印结果并退出，不执行 schedule.train()
    selected_size = len(schedule.labeled_idx[schedule.labeled_idx == True])
    rank0_print(f"*** Data Selection Done! Selected Data Size = {selected_size}")
    rank0_print(f"*** Data saved to: {args['data_path_root']}")
    rank0_print("*** Process finished without training.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help="Path to the config YAML file")
    args = parser.parse_args()

    main(config_file=args.config_file)