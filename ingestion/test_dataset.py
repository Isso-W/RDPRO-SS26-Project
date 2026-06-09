from datasets import load_dataset

dataset = load_dataset(
    "uoft-cs/cifar10"
)

print(dataset)

sample = dataset["train"][0]

image = sample["img"]

print(sample)

print(type(image))
print(image.size)
print(image.mode)
