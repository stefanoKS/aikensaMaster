# 13. 自動テストの実行

テストは**実機カメラ・GPU・PLC を必要としません**。
合成画像と一時ディレクトリだけで動作します。

## 実行方法

### Windows

```bat
tools\run_tests.cmd
```

### Linux / macOS

```bash
tools/run_tests.sh
```

### 直接実行する場合

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1      # Windows: set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest                # すべて
python -m pytest tests/test_geometry.py -v
```

> `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` が必要な理由:
> ROS 2 が入っているワークステーションでは、pytest の自動プラグイン読み込みが
> `ModuleNotFoundError: No module named 'lark'` などで失敗します。
> 本プロジェクトのテストは外部 pytest プラグインを必要としません。

## テストの構成

| ファイル | グループ | 内容 |
|----------|----------|------|
| `tests/test_machine_configuration.py` | A | 欠損 YAML、null、重複シリアル、不正解像度、再起動、設備間の分離 |
| `tests/test_calibration_and_revisions.py` | B / G | null 校正、不正データ、退化ホモグラフィ、保存拒否、版のロールバック、バックアップ検証 |
| `tests/test_geometry.py` | C | 縮小変換の一致、変換合成、回転中心、マスク、クロップ、ウィジェット分離 |
| `tests/test_camera_lifecycle.py` | D | 取得/解放、二重所有の禁止、プレースホルダ拒否、読み取り中の解放防止 |
| `tests/test_settings_ui.py` | E | `.ui` の XML 妥当性・読み込み・objectName、画面遷移、保存、ロールバック |
| `tests/test_inspection_safety.py` | F | ハードコードされたシリアルの不在、未実装検査の明示エラー、デバッグバイパス無効 |

## 画面のヘッドレステスト

UI テストは `QT_QPA_PLATFORM=offscreen` で動作します
（`tests/conftest.py` で自動設定）。ディスプレイは不要です。

## 実機でしか確認できない項目

以下は自動テストの対象外です。実機で確認してください。
詳細: [hardware_checklist.md](hardware_checklist.md)

- IC4 カメラの実接続・露光・ホワイトバランスの反映
- ChArUco ボードの実際の検出精度・再投影誤差
- ホモグラフィ算出後の結合画像の実際の見え方
- AI モデルによる実際の OK/NG 判定
- PLC (SiO) との信号やり取り

## テスト結果の見方

```
109 passed in 25.3s
```

`failed` がある場合は、まず以下を確認してください。

1. `aikensa/core/config/defaults/` が存在するか（テンプレート）
2. `PyQt5` がインストールされているか
3. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` を設定したか
