import pandas as pd

df = pd.read_parquet('/data3/peijia/dr-claw/Explain/Experiment/core_code/training/training_data/grpo/train.parquet')

has_correct = 0
no_correct = 0
no_explores = 0

for _, row in df.iterrows():
    ei = row['extra_info']
    cached = []
    try:
        cached = ei['tools_kwargs']['explore']['create_kwargs']['cached_explores']
    except (KeyError, TypeError):
        no_explores += 1
        continue

    if len(cached) == 0:
        no_explores += 1
        continue

    found = False
    for e in cached:
        v = e.get('is_correct')
        if v is True or (isinstance(v, (int, float)) and v == 1):
            found = True
            break
    if found:
        has_correct += 1
    else:
        no_correct += 1

total = len(df)
print(f"Total questions:            {total}")
print(f"No cached explores:         {no_explores}")
print(f">=1 correct explore:        {has_correct}  ({has_correct/total*100:.1f}%)")
print(f"0  correct explores (hard): {no_correct}  ({no_correct/total*100:.1f}%)")

# Show is_correct type for first row
ei = df.iloc[0]['extra_info']
try:
    cached = ei['tools_kwargs']['explore']['create_kwargs']['cached_explores']
    e0 = cached[0]
    print(f"\nSample explore keys: {list(e0.keys())}")
    print(f"is_correct type: {type(e0.get('is_correct'))}, value: {repr(e0.get('is_correct'))}")
    print(f"n explores for q0: {len(cached)}, correct: {sum(1 for e in cached if e.get('is_correct') is True or e.get('is_correct')==1)}")
except Exception as ex:
    print(f"inspect failed: {ex}")
