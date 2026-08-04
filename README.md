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

安装完成后关闭并重新打开终端，确认 Python 版本：

```cmd
py -3.13 --version
py -0p
```

推荐使用项目安装脚本创建独立虚拟环境并安装全部依赖：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

如果不使用虚拟环境，也可以直接使用 Python 3.13 安装 `requirements.txt`：

```cmd
py -3.13 -m pip install --upgrade pip
py -3.13 -m pip install --user -r requirements.txt
```

如果只运行不需要采集卡的普通 Macro，可以只安装串口依赖：

```cmd
py -3.13 -m pip install --user pyserial
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

以下命令统一使用 Windows Python Launcher 的 `py -3.13`，不要求 `.venv` 存在。如果已经通过 `setup.ps1` 成功创建虚拟环境，可以将命令开头的 `py -3.13` 替换为 `.\.venv\Scripts\python.exe`。

自动对战：

```powershell
py -3.13 .\autocontroller_rebuild_for_RL\main.py --config .\autocontroller_rebuild_for_RL\runtime_config.local.json --tmp_win_target
```

普通宏手柄（`macro1` 至 `macro999`，当前已实现的配置以代码注册表为准）：

```powershell
py -3.13 .\autocontroller_rebuild_for_RL\macro_gamepad.py --config .\autocontroller_rebuild_for_RL\runtime_config.local.json --macro macro1
```

运行期间按 `P` 暂停/恢复。暂停会释放按键和摇杆，并冻结当前动作及 macro1～5 的卖装计时；恢复时先重新执行一次手柄检测，再继续原序列。macro1～5 每运行 90 分钟，会在下一轮进入地图前执行一次卖装；macro6 不执行卖装。按 `Q` 或 `Ctrl+C` 退出。

智能宏手柄（在普通宏控制基础上增加视频状态观察）：

```powershell
py -3.13 .\autocontroller_rebuild_for_RL\smart_macro_gamepad.py --config .\autocontroller_rebuild_for_RL\runtime_config.local.json
```

克隆水母对战：

```powershell
py -3.13 .\autocontroller_rebuild_for_RL\clone_jelly_main.py --config .\autocontroller_rebuild_for_RL\runtime_config.local.json
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

## 附加

本项目做了一个简陋的终端展示，可以自行进行局域网联机占地斗士对战游玩，但是需要自行根据卡牌编号，配置对应的卡牌文件：

启用局域网服务端：

```powershell
py -3.13 .\tableturf_sim\tools\play_service.py --bind 0.0.0.0
```

占地斗士启动客户端（主机创建房间需要运行服务端）：

```powershell
py -3.13 .\tableturf_sim\tools\play_client.py --name Host
```

占地斗士启动客户端简易客机端：

```powershell
py -3.13 .\tableturf_sim\tools\play_client_simple.py --name Client
```

相关卡牌/牌组文件：

- `tableturf_sim/tools/play_client_simple_decks.json`
- `tableturf_sim/data/cards/PlayerPresetDeck.json`

工具：

视频流展示（即流程3中调用的）：

```powershell
py -3.13 .\vision_capture\preview_stream_opencv.py
```

## 无用废话
本项目存在大量临时文件，未进行整理。
目前在我自己的本机上是能用，但是未进行其它设备的尝试，如果有任何bug请随时与我联系！
