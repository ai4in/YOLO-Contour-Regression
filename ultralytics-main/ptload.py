import  torch
loaded_tensor = torch.load('/home/minxuanlin/ultralytics/ultralytics-main/tensor_example15.pt')
print(loaded_tensor.shape)


# 指定保存文件的路径
save_path = '/home/minxuanlin/ultralytics/ultralytics-main/ultralytics/tensor_example15.txt'
# 将张量的值写入文本文件

print("xx")
with open(save_path, 'w') as file:
    for value in loaded_tensor:
        for v in value.view(-1):
            file.write(f'{v.item()}\n')

