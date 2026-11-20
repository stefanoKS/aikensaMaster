# 10. `git pull` から設備設定を守る

設備ごとに異なるファイルは、**Git の管理対象から外してあります**。

| ファイル | Git 管理 | 備考 |
|----------|----------|------|
| `aikensa/core/config/cam/cam_id.yaml` | 対象外 | カメラのシリアル番号 |
| `aikensa/core/config/cam/cam_config.yaml` | 対象外 | 解像度・露光など |
| `aikensa/core/param/camera/*` | 対象外 | 校正・ホモグラフィ・平面化 |
| `aikensa/core/param/adjustments/*` | 対象外 | 微調整・履歴 |
| `aikensa/core/config/defaults/*` | **管理対象** | `null` のテンプレート |

## なぜ `.gitignore` だけでは不十分なのか

Git に**すでに登録されている**ファイルは、`.gitignore` に追加しても追跡が解除されません。
追跡解除（`git rm --cached`）を行うコミットを他の設備が `git pull` すると、
そのファイルは作業ツリーから**削除されます**。

つまり「追跡解除のコミットを配布する」ときだけが危険ポイントです。
そのため、必ず次の手順を守ってください。

## 既存設備での必須手順（`git pull` の前に実行）

```bat
:: 1. AIKensaMaster を終了する
:: 2. バックアップを取得し、検証されることを確認する
tools\backup_machine_config.cmd
        ↓
   「RESULT: BACKUP VERIFIED (SHA-256 checksums match)」を確認
        ↓
:: 3. ここで初めて git pull する
git pull
        ↓
:: 4. 設定が残っているか確認する
python tools\machine_config_backup.py status
:: 5. 消えていた場合は復元する
tools\restore_machine_config.cmd
```

### バックアップが "BACKUP VERIFIED" にならない場合

**`git pull` を行わないでください。** 原因を解消してから再実行します。

- ディスク容量不足
- `AIKENSA_BACKUP_DIR` の権限不足
- バックアップ先がリポジトリ内になっている

## 新しい設備（新規クローン）の場合

新規クローンでは設備固有ファイルが存在しないため、起動時に
`null` のテンプレートから自動生成されます。
**他の設備のシリアル番号や校正値が引き継がれることはありません。**

```bat
git clone https://github.com/stefanoKS/aikensaMaster.git
cd aikensaMaster
python main.py     :: 起動時に null テンプレートが作成される
```

起動後、「設備設定」→「カメラ設定」で自設備のシリアル番号を設定してください。

## 日常運用でのチェック

```bat
python tools\machine_config_backup.py status
```

出力例:

```
application root        : C:\aikensaMaster
machine files present   : 24
still tracked by Git    : 0
backups found           : 3
backups verified        : 3
untracking safe         : YES
```

- `still tracked by Git` が 0 件であること
- `backups verified` が 1 件以上あること

## やってはいけないこと

- `git reset --hard` / `git clean -fdx`（設備設定が消えます）
- `git checkout -- aikensa/core/param/camera/`（校正値が巻き戻ります）
- 設備固有ファイルを `git add -f` すること
- バックアップを確認せずに追跡解除コミットを他設備へ配布すること

詳細な移行手順は [git_migration_runbook.md](git_migration_runbook.md) を参照してください。
