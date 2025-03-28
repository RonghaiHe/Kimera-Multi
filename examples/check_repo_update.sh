#!/bin/bash

# 检查.repos文件是否存在
if [ ! -f kimera_multi.repos ]; then
  echo ".repos 文件不存在"
  exit 1
fi

# 获取当前目录的父目录路径
current_dir=$(pwd)
parent_dir=$(dirname "$current_dir")

# 初始化变量
repo_name=""
repo_url=""
repo_version=""

# 读取.repos文件中的仓库信息
while IFS= read -r line; do
  # 跳过空行和注释行
  if [[ -z "$line" || "$line" =~ ^# ]]; then
    continue
  fi

  # 跳过repositories:这一行
  if [[ "$line" =~ ^repositories: ]]; then
    continue
  fi

  # 解析仓库名称
  if [[ "$line" =~ ^[[:space:]]*([a-zA-Z0-9_-]+):[[:space:]]*$ ]]; then
    repo_name="${BASH_REMATCH[1]}"
    continue
  fi

  # 解析仓库URL
  if [[ "$line" =~ ^[[:space:]]*url:[[:space:]]*(.+) ]]; then
    repo_url="${BASH_REMATCH[1]}"
    continue
  fi

  # 解析仓库版本
  if [[ "$line" =~ ^[[:space:]]*version:[[:space:]]*(.+) ]]; then
    repo_version="${BASH_REMATCH[1]}"
    # 当解析到版本信息时，处理当前仓库
    if [ -n "$repo_name" ] && [ -n "$repo_url" ]; then
      # 构造仓库路径
      repo_path="$parent_dir/$repo_name"

      # 检查仓库路径是否存在
      if [ ! -d "$repo_path" ]; then
        echo "仓库路径 $repo_path 不存在"
        continue
      fi

      # 进入仓库目录
      cd "$repo_path"

      # 获取当前跟踪的远程仓库
      current_remote=$(git remote get-url origin)

      # 检查是否跟踪到了对应的远程仓库
      if [ "$current_remote" != "$repo_url" ]; then
        echo "仓库 $repo_path 未跟踪到对应的远程仓库"
        echo "当前远程仓库: $current_remote"
        echo "期望的远程仓库: $repo_url"

        # 切换到对应的远程仓库
        git remote set-url origin "$repo_url"
        echo "已切换远程仓库"
      else
        echo "仓库 $repo_path 已跟踪到对应的远程仓库"
      fi

      # 检查当前分支是否为期望的版本
      current_branch=$(git rev-parse --abbrev-ref HEAD)
      if [ "$current_branch" != "$repo_version" ]; then
        echo "仓库 $repo_path 当前分支不是期望的版本"
        echo "当前分支: $current_branch"
        echo "期望的版本: $repo_version"

        # 切换到期望的版本
        git checkout -B "$repo_version"
        echo "已切换到版本 $repo_version"
      else
        echo "仓库 $repo_path 当前分支已是期望的版本"
      fi

      git pull
      echo "已拉取最新代码"
      git branch --set-upstream-to="origin/$repo_version" "$repo_version"
      echo "已设置跟踪分支"

      # 返回到脚本所在目录
      cd -
    fi

    # 重置变量以便解析下一个仓库
    repo_name=""
    repo_url=""
    repo_version=""
  fi
done < kimera_multi.repos