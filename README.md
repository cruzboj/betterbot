# Project - BetterBot

run dockerfile:
```
docker compose up -d
```

to enter jupyter server http://localhost:8888


or check genarated jupter url:
```
docker logs betterbot_container
```


requires-python = "==3.12.*"

for local run GPU users , cu130 50serise , cu126 40serise
```
[X] uv add --index https://download.pytorch.org/whl/cu124 torch torchvision torchaudio 
[X] uv add --index https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
uv add --index https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
```



