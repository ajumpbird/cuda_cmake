# 介绍

记录cuda学习成果，包括：

1. 用cuda实现扇束fbp算法
2. 用cuda实现卷积运算，对图像进行滤波

使用了cmake，cuda为c++版。我使用的是vscode编辑器，为了成功运行代码，需要进行一些必要的配置，比如cuda、opencv的路径，当然，这些代码可以移植到visual studio中运行。

```shell
cd .\01_fan_fbp\
cmake -B build -S .
cmake --build .\build\
.\build\Debug\app.exe
```

> 由于不同文件夹下的include文件夹包含同名文件，在运行某个文件夹中的代码时，最好将c_cpp_properties.json中的其他include注释掉，防止vscode报错，但这并不是代码本事错误，仍可以运行。
