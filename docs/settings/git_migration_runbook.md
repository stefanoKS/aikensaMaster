# Git 移行手順（STAGE A / STAGE B）

このドキュメントは**エンジニア向け**です。運用者向けは
[git_pull_protection.md](git_pull_protection.md) を参照してください。

## 背景

以前のリポジトリでは、設備固有のファイルが Git に**追跡**されていました。

```
aikensa/core/config/cam/cam_id.yaml
aikensa/core/config/cam/cam_config.yaml
aikensa/core/param/camera/Calibration_camera_0.yaml
aikensa/core/param/camera/Calibration_camera_1.yaml
aikensa/core/param/camera/Calibration_camera_scaled_0.yaml
aikensa/core/param/camera/Calibration_camera_scaled_1.yaml
aikensa/core/param/camera/homography_param_cam1.yaml
aikensa/core/param/camera/homography_param_cam1_scaled.yaml
aikensa/core/param/camera/homography_param_cam2.yaml
aikensa/core/param/camera/homography_param_cam2_scaled.yaml
aikensa/core/param/camera/planarizeTransform_wide.yaml
aikensa/core/param/camera/planarizeTransform_wide_scaled.yaml
```

`.gitignore` を追加するだけでは追跡は解除されません。
追跡解除のコミットを既存設備が `git pull` すると、**作業ツリーからファイルが削除されます。**

## STAGE A — 準備（安全側の作業）

実施済みの内容:

1. **テンプレートを Git 管理下に追加**
   `aikensa/core/config/defaults/` に `null` のテンプレートを配置。
   - `cam/cam_id.yaml` … `0: null` / `1: null`
   - `cam/cam_config.yaml`
   - `camera/Calibration_camera_{0,1}[_scaled].yaml`
   - `camera/homography_param_cam{1,2}[_scaled|_high|_high_scaled].yaml`
   - `camera/planarizeTransform_*.yaml`
   - `adjustments/global_alignment.yaml` / `widget_adjustments.yaml`

2. **起動時の安全な初期化を実装**
   `SettingsRepository.ensure_runtime_layout()` は**存在しないファイルだけ**を作成します。

3. **バックアップ／復元ツールを実装**
   - `tools/machine_config_backup.py`（`backup` / `verify` / `restore` / `status`）
   - `tools/backup_machine_config.cmd`（Windows）
   - `tools/verify_machine_config_backup.cmd`
   - `tools/restore_machine_config.cmd`
   - POSIX 版 `.sh` も同梱

4. **`.gitignore` に実パスを明示**

5. **実データでのバックアップ検証**
   開発機で 24 ファイルをバックアップし、SHA-256 検証に成功。

## STAGE B — 追跡解除

実施内容（このリポジトリの該当コミット）:

```bash
git rm --cached \
  aikensa/core/config/cam/cam_id.yaml \
  aikensa/core/config/cam/cam_config.yaml \
  aikensa/core/param/camera/Calibration_camera_0.yaml \
  aikensa/core/param/camera/Calibration_camera_1.yaml \
  aikensa/core/param/camera/Calibration_camera_scaled_0.yaml \
  aikensa/core/param/camera/Calibration_camera_scaled_1.yaml \
  aikensa/core/param/camera/homography_param_cam1.yaml \
  aikensa/core/param/camera/homography_param_cam1_scaled.yaml \
  aikensa/core/param/camera/homography_param_cam2.yaml \
  aikensa/core/param/camera/homography_param_cam2_scaled.yaml \
  aikensa/core/param/camera/planarizeTransform_wide.yaml \
  aikensa/core/param/camera/planarizeTransform_wide_scaled.yaml
```

`--cached` により**作業ツリーのファイルは残ります**。

## デプロイ（既存設備）— 必須ゲート

> **このコミットを既存設備に配布する前に、各設備で STAGE A のバックアップを
> 取得し、検証済みであることを確認してください。**
> 確認できない場合、デプロイは **BLOCKED** です。

各設備での手順:

```bat
:: 0. AIKensaMaster を終了
:: 1. 現在の追跡状態と値を確認
python tools\machine_config_backup.py status
:: 2. バックアップ（検証まで）
tools\backup_machine_config.cmd
::    "RESULT: BACKUP VERIFIED" を確認
:: 3. ここで初めて pull
git pull
:: 4. 設定が残っているか確認
python tools\machine_config_backup.py status
::    still tracked by Git: 0
::    backups verified: 1 以上
:: 5. 万一消えていたら復元
tools\restore_machine_config.cmd
```

`git pull` 後にファイルが消えていた場合は、**STAGE A のバックアップから復元**してください
（新規クローンと同じ `null` 状態に戻す必要はありません）。

## 新規クローン

```bat
git clone https://github.com/stefanoKS/aikensaMaster.git
cd aikensaMaster
python main.py
```

- `cam_id.yaml` は `0: null` / `1: null` で生成されます
- 校正ファイルは `null` で生成されます
- 他設備のシリアル番号・校正値は**一切含まれません**

## 検証コマンド

```bash
python tools/machine_config_backup.py status
# untracking safe : YES
```

`untracking_safe` は以下を満たすときに `YES` になります。

1. `cam_id.yaml` 等が Git に追跡されていない
2. 検証済みバックアップが 1 件以上存在する
