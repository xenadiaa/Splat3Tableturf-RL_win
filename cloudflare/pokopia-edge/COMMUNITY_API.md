# 玩家开门信息与好人榜

该功能完全运行在 Cloudflare Worker + D1。Windows/Macro6 离线时，公开查询、提交、反馈和好人榜仍可使用。

## 公开接口

- `GET /api/community/rooms`：12小时内且未隐藏的开门信息。
- `POST /api/community/rooms`：发布开门信息。
- `POST /api/community/rooms/{id}/feedback`：`full`、`full_wrong`、`invalid`、`invalid_wrong`；其中已满反馈只接受梦幻章车。
- `DELETE /api/community/rooms/{id}`：原发布浏览器凭本地 `owner_token` 删除。
- `GET /api/community/leaderboard?period=day|week|month|year|all&player=模糊名称`：好人榜，支持自定义`start/end`业务日期范围；每名玩家同时返回总开门次数和梦幻章车、任务车、花车、材料单车、其它车的分类次数。
- `GET /api/community/status`：公开写入开关状态。

信息超过12小时或确认失效超过30分钟后只是不再公开展示，D1记录不会定时物理删除。发布者主动纠错删除的记录不会进入好人榜。

发布开门码为必填字段，必须正好包含6位数字或英文字母；可以全数字、全字母或混合。车种支持梦幻章车、任务车、花车、材料单车和其它车。

## 防滥用

- 所有查询和写操作按接口限制为每地址每秒一次。
- 同一来源网络每天最多发布30条、反馈100次，同时最多保留5条有效车辆；连续发布间隔至少15秒。
- 同一来源网络10分钟内不能对同一车辆重复相同反馈，但可以反馈多辆不同车辆。
- 发布不足5分钟的新车需要两个不同来源网络确认，才会标为失效；第一次显示“失效待核验”。
- 后台只保存由部署令牌加盐的IP哈希，并按业务日累计上传、反馈、早期失效反馈次数；不保存明文IP。

## 管理员接口

使用与Macro6边缘上传相同的Bearer令牌鉴权：

- `POST /api/admin/community/toggle`：切换或指定公开提交/反馈开关。
- `GET /api/admin/community/export`：下载房间、反馈、用量和设置数据。
- `POST /api/admin/community/import`：合并上传`pokopia-community-backup-v1`备份。

Windows快捷操作：

- 双击根目录 `toggle_pokopia_community_writes.bat` 切换总开关。
- `python autocontroller_rebuild_for_RL\pokopia_community_admin.py export`
- `python autocontroller_rebuild_for_RL\pokopia_community_admin.py import 备份.json --confirm-import YES`
