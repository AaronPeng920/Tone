# 简明 Git 教程

## 介绍

> git 是一个**分布式**(本地和远程)**版本控制**系统

## 本地操作

1. 新建一个项目文件夹 `GitTest`
2. 在项目文件夹中完成自己的项目，这里用 Hello World 为例：

![image-20230424172052073](/Users/aaronpeng/Library/Application Support/typora-user-images/image-20230424172052073.png)

3. 初始化 git 仓库，此项目本质上只是本地的一个文件夹，如果想让它成为一个 Git 仓库，在此项目下执行 `git init`：

![image-20230424172219463](/Users/aaronpeng/Library/Application Support/typora-user-images/image-20230424172219463.png)

> 执行 `git init` 后，`.git` 目录在根目录下被创建，其包括所有当前 git 仓库所需要的所有信息。除去 `.git` 目录之外，其它任何目录部分都没有任何改变，也就是说即使该文件夹下之前有文件，这些文件都还没有加入到 git 仓库中
>
> `.git` 目录下含有以下的内容：
>
> ![image-20230424172554717](/Users/aaronpeng/Library/Application Support/typora-user-images/image-20230424172554717.png)

4. 向 git **暂存区**中添加新的文件，执行 `git add 文件/路径/RE统配`，如果指定目录，那么目录下的所有文件都会加入到 git 暂存区中，本例子中执行 `git add .`，将项目的所有文件都加入到暂存区中

> 注意，本地的项目文件夹叫作工作区，当执行 `git add` 命令时，暂存区的目录树被更新，同时工作区修改(或新增)的文件内容被写入到对象库中的一个新的对象中，而该对象的 ID 被记录在暂存区的文件索引中。
>
> ![img](https://www.runoob.com/wp-content/uploads/2015/02/1352126739_7909.jpg)

5. 提交文件，将文件真正的保存在 git 仓库中，使用 `git commit -m "提交信息"`



## 远程仓库

1. 在 github 上创建一个新的空的远程仓库，该仓库的默认分支是 `main`

![image-20230424174107630](/Users/aaronpeng/Library/Application Support/typora-user-images/image-20230424174107630.png)

2. 在本地仓库中添加远程仓库，使用 `git remote add 名字 URL`，这里的名字其实就是远程仓库的别名，往往默认使用 `origin`，在通过 `git clone` 的时候默认的远程仓库的别名也是 `origin`，这里我使用的是 `yuancheng`

3. 推送本地仓库到远程，使用 `git push 远程仓库名 远程仓库分支名`，如我的命令是 `git push yuancheng main`，有时候遇到错误可以使用强推 `git push yuancheng main`







