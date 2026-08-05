# 前言

本目录用于 Windows 适配整理。目前仍有功能未完全适配，但安装和启动入口已按 Windows 方式整理。

如果您使用我的项目进行自动对战，生成了replay文件(Splat3Tableturf-RL/autocontroller_rebuild_for_RL/replays)，希望能够打包邮件发送给我！
glibz@connect.ust.hk
不胜感激！


# Splat3Tableturf-RL使用

## 1. 获取库

```bash
git clone https://github.com/xenadiaa/Splat3Tableturf-RL_win
cd Splat3Tableturf-RL_win
```

## 2. 执行安装

推荐先在 CMD 或 PowerShell 中安装 Python 3.13：

```cmd
winget install -e --id Python.Python.3.13
```

安装完成后关闭并重新打开终端，确认 `python` 已指向 Python 3.13：

```cmd
python --version
where python
```

本项目优先推荐直接使用系统 Python 安装依赖，不要求创建虚拟环境：

```cmd
python -m pip install --upgrade pip
python -m pip install --user -r requirements.txt
```

如果电脑中安装了多个 Python，且 `python --version` 不是 3.13，可以临时使用 Python Launcher 指定版本：

```cmd
py -3.13 -m pip install --user -r requirements.txt
```

虚拟环境安装脚本仍然保留，但只作为可选方式：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

如果只运行不需要采集卡的普通 Macro，可以只安装串口依赖：

```cmd
python -m pip install --user pyserial
```

自动对战和智能 Macro 需要 FFmpeg。推荐使用 WinGet 安装：

```cmd
winget install -e --id Gyan.FFmpeg
```

安装完成后需要完全关闭当前 CMD/PowerShell 并重新打开，然后检查：

```cmd
where ffmpeg
ffmpeg -version
```

如果仍然无法识别 `ffmpeg`，先确认软件已经安装：

```cmd
winget list --id Gyan.FFmpeg
```

如果能够查到 FFmpeg 但命令仍不可用，请重启 Windows 以刷新 PATH。普通 `macro_gamepad.py` 不使用视频流，因此不需要 FFmpeg 或采集卡。

也可以使用其他包管理器安装：

```powershell
choco install ffmpeg
scoop install ffmpeg
conda install -c conda-forge ffmpeg
```

## 3. 运行主命令

以下命令均从项目根目录执行，并直接使用系统 `python`。请先通过 `python --version` 确认版本为 Python 3.11 以上，推荐 Python 3.13。

自动对战：

```powershell
python .\autocontroller_rebuild_for_RL\main.py --config .\autocontroller_rebuild_for_RL\runtime_config.local.json --tmp_win_target
```

普通宏手柄（`macro1` 至 `macro999`，当前已实现的配置以代码注册表为准）：

```powershell
python .\autocontroller_rebuild_for_RL\macro_gamepad.py --config .\autocontroller_rebuild_for_RL\runtime_config.local.json --macro macro1
```

运行期间按 `P` 暂停/恢复。暂停会释放按键和摇杆，并冻结当前动作及 macro1～5 的卖装计时；恢复时先重新执行一次手柄检测，再继续原序列。macro1～5 每运行 90 分钟，会在下一轮进入地图前执行一次卖装；macro6 不执行卖装。按 `Q` 或 `Ctrl+C` 退出。

智能宏手柄（在普通宏控制基础上增加视频状态观察）：

```powershell
python .\autocontroller_rebuild_for_RL\smart_macro_gamepad.py --config .\autocontroller_rebuild_for_RL\runtime_config.local.json --macro macro1
```

`smart macro1` 沿用 macro5 与每 90 分钟卖装流程。手柄检测后，第一轮前执行 Y、A、等待 10 秒和摇杆后 5 秒；第一轮持续重复“额外 A 4 秒 → 进入地图 → A 4 秒+A×3”，并在整个序列期间持续检测画面，命中四个白色图标后直接进入 ZR 阶段。第二轮起每轮先进入地图一次，再重复“A 4 秒+A×3检测”直到命中。第一轮大循环与第二轮小循环都会在每个按键前、按住期间及按键间隔持续检测，命中后立即释放按键并跳过剩余序列。射击阶段持续保持 ZR，并按 L→R→A 轮换，每 1 秒短按一个技能键、每 3 秒完成一套。

`smart macro2` 在手柄检测后、第一轮前短按 A 四次（每次 50ms、间隔 500ms），等待 10 秒并将左摇杆向后推动 5 秒；不再执行选图序列，第一轮及后续轮次均直接循环“A 4 秒+A×3检测”直到命中。长按 A、短按 A1、A2、A3 每个按键发送前都会检测，命中后立即跳过当前及剩余按键并进入 ZR 阶段。射击阶段持续保持 ZR，并按 L→R→A 轮换，每 1 秒短按一个技能键、每 3 秒完成一套。

智能宏支持与占地斗士终端一致的手动键盘控制：`Z=A`、`X=B`、`A=Y`、`S=X`、方向键控制左摇杆、`C=L`、`V=R`、`F=ZL`、`G=ZR`、`+`/`=`/`D=Plus`、`-`/`E=Minus`。`P` 暂停或恢复宏，`Q` 退出；暂停期间仍可使用手动控制。键盘操作按终端短按脉冲发送，不依赖全局键盘监听库。

克隆水母对战：

```powershell
python .\autocontroller_rebuild_for_RL\clone_jelly_main.py --config .\autocontroller_rebuild_for_RL\runtime_config.local.json
```

启动前说明：

- 需要在《斯普拉遁 3》的占地斗士中，提前配置好所需使用的卡牌组，并放置在左上角位置
- 不同卡牌组对不同地图的胜率会有差异
- 在选择好 NPC 难度、进入对战并来到选卡界面后，再启动代码（即，进入到只需按A即可进入对战的界面，比brianuuu的宽容度高，允许在选好难度后，按A位置进入，不会出现误识别对战状态）

启动后说明：
- 首次启动会进入串口选择，使用brianuuu/AutoController的智能固件烧录的Arduino or other device.
- 首次进入会进入视频流选择界面，选择对应采集卡即可，如选择错误，可在视频流界面按“R”，进入重选
- 其余操作快捷键见启动后终端以及视频流窗口，其中可手动进行暂停/与智能设备共同控制按键（包含度不高，只包含占地斗士可用键：DPad、ABXY、+、Home、L

## 4. 对战配置

- `autocontroller_rebuild_for_RL/runtime_config.local.json`
  - 自动对战配置，可设置对战需求或使用的策略网络


## 5. 策略训练

目前有PPO训练网络，可自行研究使用
先前自对弈策略训练结果未上传，因为效果远不如当前使用策略，后续数据充足重新训练，如果效果好会上传，该部分需大家合力而为，我个人获取的replay数据是有限的，为了充足的数据支撑策略训练，还望大家能够将回放文件发予我！

## 6. 视频流工具

自动对战会自动启动视频流。需要单独测试采集卡、帧率或 Frame API 时，可以在项目根目录运行：

```cmd
python .\vision_capture\preview_stream_opencv.py
```

启动后默认提供：

- 视频帧接口：`http://127.0.0.1:8765/frame.jpg`
- 健康状态接口：`http://127.0.0.1:8765/health`
- `Enter`：保存当前截图
- `B`：连续保存 30 帧
- `R`：重新选择视频设备
- `Esc` 或 `Ctrl+C`：退出

视频设备选择会写入 `vision_capture/capture_config.json`。程序会枚举 Windows 中的全部视频设备，疑似采集卡只会优先显示，不会排除其他品牌。

## 7. 局域网占地斗士

本项目做了一个简陋的终端展示，可以自行进行局域网联机占地斗士对战游玩，但是需要自行根据卡牌编号，配置对应的卡牌文件：

服务端和客户端需要分别在独立的 CMD/PowerShell 窗口中运行。首次启动服务端时，如果 Windows 防火墙询问是否允许 Python 访问网络，请允许专用网络访问。

启用局域网服务端：

```powershell
python .\tableturf_sim\tools\play_service.py --bind 0.0.0.0
```

占地斗士启动客户端（主机创建房间需要运行服务端）：

```powershell
python .\tableturf_sim\tools\play_client.py --name Host
```

占地斗士启动客户端简易客机端：

```powershell
python .\tableturf_sim\tools\play_client_simple.py --name Client
```

相关卡牌/牌组文件：

- `tableturf_sim/tools/play_client_simple_decks.json`
- `tableturf_sim/data/cards/PlayerPresetDeck.json`

## 无用废话
本项目存在大量临时文件，未进行整理。
目前在我自己的本机上是能用，但是未进行其它设备的尝试，如果有任何bug请随时与我联系！
