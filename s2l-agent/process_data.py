from datasets import load_dataset

# 1. 加载数据集 (假设是本地json文件，或是HF上的路径 'TIGER-Lab/MathInstruct')
dataset = load_dataset('/home/pjp/S2L-agent/MathInstruct', split='train')
# 或者加载本地: dataset = load_dataset('json', data_files='your_data.json', split='train')

# 2. 随机打乱并选取前1万条
# seed=42 保证每次运行选出的数据是一样的，方便复现
sampled_dataset = dataset.shuffle(seed=42).select(range(10000))

# 3. 保存结果
sampled_dataset.to_json('MathInstruct.jsonl')

print(f"原数据量: {len(dataset)}, 采样后数据量: {len(sampled_dataset)}")