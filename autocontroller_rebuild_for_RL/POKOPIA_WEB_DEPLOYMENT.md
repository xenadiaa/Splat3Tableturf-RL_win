# Pokopia 梦幻章状态页部署说明（rabi.date）

## 最终网络结构

正式方案不使用家庭公网IP、路由器端口映射、Cloudflare Tunnel或动态DNS：

`Macro6 → 本机只读网页(127.0.0.1:8787) → 独立上传进程 → HTTPS → Cloudflare边缘`

访问者只连接Cloudflare：

`访客 → stamp.rabi.date → Worker + Durable Object + D1 + R2`

- `rabi.date` 永久重定向到 `https://stamp.rabi.date`，并保留路径和查询参数；
- 根首页会优先进入Worker执行跳转，其他CSS/JS/图案静态资源仍由Assets直接缓存；
- Windows主动上传公开状态，家庭公网IP不进入DNS，也不开放任何路由器端口；
- 实时状态写入Durable Object，并通过WebSocket主动推送给在线页面；
- 浏览器倒计时每秒在本地推进，不会每秒查询服务器；
- 最新CODE截图固定覆盖R2的 `current-code.png`，不能枚举历史截图；
- 每日永久统计进入D1，用于日/周/月/年/全部范围和玩家名模糊查询；
- 原始执行日志、ERROR图、姓名OCR图、F10图、结束图、Chrome 9222和串口均不会公开；
- Windows上传令牌由当前Windows用户的DPAPI加密，配置文件本身不能在另一台电脑/用户下解密。

## 访问频率与相互隔离

实时页只维持一条WebSocket；连接断开时才以10秒一次的GET作为降级读取。本机预览仍每2秒读取。
历史查询和排行榜以 `访客IP + 浏览器会话ID + 接口种类` 为键，每个接口每秒最多一次；页面也会先在
浏览器内排队，因此不同访客、不同浏览器标签页的筛选内容不会互相覆盖。浏览器会话ID不是安全凭证，
正式公开后还应在Cloudflare为 `/api/query*` 和 `/api/rankings*` 加一条IP级限流/WAF规则，阻止恶意轮换ID。

## 首次部署前只需确认一次

1. 登录Cloudflare，确认账户首页能看到 `rabi.date`，状态为 **Active**；
2. 如果域名不是在Cloudflare Registrar购买：先在Cloudflare选择“添加域”，再到注册商把名称服务器改成
   Cloudflare分配的两条；切换前按Cloudflare提示处理旧DNSSEC，等待状态Active；
3. DNS中不能已有占用 `rabi.date` 或 `stamp.rabi.date` 的A/AAAA/CNAME记录。只删除冲突的网页记录，
   不要删除邮件用MX/TXT记录；
4. 在运行Macro6的Windows电脑安装Node.js LTS：

   `winget install OpenJS.NodeJS.LTS`

5. 安装结束后关闭旧CMD并重新打开，使 `node`、`npm`、`npx` 进入PATH。

## Windows傻瓜式部署

双击仓库根目录的 `deploy_pokopia_cloudflare.bat`。脚本会依次：

1. 安装固定在项目目录内的Wrangler依赖；
2. 打开浏览器，让你登录并授权Cloudflare；
3. 创建或复用D1数据库 `pokopia-stamp`；
4. 创建或复用R2存储桶 `pokopia-stamp-media`；
5. 应用数据库结构并部署Worker；
6. 为上传接口生成高强度秘密令牌；
7. 将令牌写入Cloudflare Secret，并用Windows DPAPI保存本机副本；
8. 建立 `rabi.date` 与 `stamp.rabi.date` 两个Custom Domain。

BAT会另开一个不会自动关闭的部署窗口。Wrangler登录调用Windows系统默认浏览器，并不固定调用Chrome；
如果默认浏览器是Chrome，就会打开Chrome。若部署仍失败，窗口会保留报错，并在仓库根目录生成
`pokopia_cloudflare_deploy_error.log`。

脚本可重复执行；它会复用现有D1、R2和本机令牌，不会重新导入或叠加历史统计。

部署成功后，正常双击 `run_pokopia_web_and_watchdog.bat`。启动器会自动启动三个互相隔离的进程：

- 本机只读网页；
- Cloudflare边缘上传器（仅在检测到加密配置时启动）；
- 原Smart Macro6 watchdog。

边缘断网只会使上传器退避重试，Macro6不会因此退出或暂停。状态心跳超过30秒后，公开页自动显示离线。

## Cloudflare面板上线后检查

依次验证：

1. `https://stamp.rabi.date/health` 返回 `ok: true`；
2. `https://rabi.date/任意路径?a=1` 跳到 `https://stamp.rabi.date/任意路径?a=1`；
3. Windows启动Macro6后，CODE、截图、房间玩家、任务数和计时器发生变化；
4. 关闭Windows启动窗口，30秒后公开页显示离线；
5. 两台设备分别做玩家查询，条件和结果互不覆盖；同一标签页连续点击不会超过每接口每秒一次；
6. Cloudflare Analytics中观察请求量。免费Workers当前有每日请求额度，访问量接近额度时再决定是否升级。

建议额外开启Cloudflare托管规则，并为上传路径 `/api/publish/*` 设置更严格的速率限制。上传接口还要求
不可公开的Bearer Token；即使有人知道URL也不能改写页面。不要把上传令牌粘贴到聊天、网页或截图中。

## 本地图片与日志保留

本机每天自动清理超过30天且文件名不含 `ERROR` 的PNG/JPG/JPEG/WEBP，包括普通CODE、开始/结束和
姓名截图。ERROR诊断图保留；JSON、JSONL、TXT日志永久保留。边缘R2只保存当前一张CODE图，下一张到来
立即覆盖。按照当前要求不创建备份，也不实现磁盘不足或系统损坏恢复。

## 仍需人工完成的事项

- 确认 `rabi.date` 在Cloudflare中为Active（代码无法替你完成注册商名称服务器变更）；
- 在Windows安装Node.js LTS并运行一次部署BAT；
- 部署后按上面的六项做一次公网验收；
- 公开运行后观察免费额度和误报，再决定WAF/IP限流的具体阈值。

除此之外，不再需要安装 `cloudflared`、配置路由器、申请公网IP、部署Java服务器或开放Windows防火墙端口。
