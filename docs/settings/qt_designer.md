# Qt Designer での `.ui` ファイル編集

## 対象ファイル

```
aikensa/gui/pages/SETTINGS/settings_main.ui          設備設定（ハブ）
aikensa/gui/pages/SETTINGS/camera_settings.ui        カメラ設定
aikensa/gui/pages/SETTINGS/calibration_settings.ui   校正・位置調整
aikensa/gui/pages/SETTINGS/fine_adjustment.ui        全体微調整
aikensa/gui/pages/SETTINGS/widget_adjustment.ui      品番別設定
aikensa/gui/pages/SETTINGS/inspection_settings.ui    検査条件
aikensa/gui/pages/SETTINGS/model_settings.ui         AIモデル設定
aikensa/gui/pages/SETTINGS/settings_history.ui       設定履歴
aikensa/gui/mainPage.ui                              メイン画面（設備設定ボタン）
```

## インストール

```bash
pip install pyqt5-tools
```

起動:

```bash
# Windows
pyqt5-tools designer
# または
"C:\<conda>\envs\aikensa\Scripts\designer.exe"
```

`.ui` ファイルは Designer の「ファイル → 開く」で直接編集できます。

## 変更してはいけないもの

| 項目 | 理由 |
|------|------|
| `aikensa/gui/pages/` の既存検査ページの順序 | `UI_FILES` の index 0〜49 が検査ロジックと対応 |
| 既存の `objectName`（`P1label` など） | `AIKensa.py` が名前で検索している |
| `settings_main.ui` の `settingsContentStack` の型 | `QStackedWidget` である必要がある |
| `camera_settings.ui` の `camera0*/camera1*` 名 | コントローラが名前で取得している |

## 追加する場合の命名規則

- カメラ 0 系: `camera0` + 役割（例 `camera0ExposureSpinBox`）
- カメラ 1 系: `camera1` + 役割
- ボタン: `<動作>Button`（例 `saveCameraSettingsButton`）
- ラベル: `<内容>Label`

コントローラ側は `findChild(QWidget, "名前")` で取得するため、
`objectName` を変更した場合は
`aikensa/gui/settings_pages/*.py` も同時に修正してください。

## レイアウトの方針

- 絶対座標（`geometry`）ではなく**レイアウト**を使用してください
  （`QVBoxLayout` / `QHBoxLayout` / `QGridLayout` / `QFormLayout`）。
- 自動テストが「レイアウトを使用していること」を検証します。
- 主要サイズ: 1920x1080 と 1280x1024 で崩れないことを確認してください。
- 文字サイズは 15〜17px を目安にします（現場の視認性）。
- メイン・製品検査・日常点検は 1920x1080 の固定デザインで、ページ全体の
  スクロールや縮小表示はありません。論理解像度が小さい画面では一部が切れます。
  設備設定の各フォームは引き続きスクロールできます。
- 表示プレビューだけを枠内に縦横比を維持して縮小します。検査用の画像は変更しません。

起動時には対象ディスプレイの論理サイズ、座標、デバイスピクセル比を
`aikensa/logs/crash.log` の Qt メッセージに記録します。
Windows の実画面、タッチ操作、カメラ表示は別途実機で確認してください。

## 変更後の検証

```bat
:: 1. XML と PyQt5 での読み込みを検証
python -m pytest tests\test_settings_ui.py -v

:: 2. 実際に起動して表示を確認
python main.py
```

`test_settings_ui.py` は次を検証します。

- XML として妥当か
- `loadUi` で読み込めるか
- 必須 `objectName` が存在するか
- レイアウトを使用しているか（絶対配置の混入チェック）
- 既存検査ページの index が変わっていないか

## よくあるエラー

| エラー | 原因 | 対処 |
|--------|------|------|
| `TypeError: setStretch(...) not enough arguments` | レイアウトに `<property name="stretch">` を書いた | 削除する（Designer では発生しません） |
| `widget X missing from ...` | `objectName` を変更した | コントローラも更新する |
| 画面が真っ白 | ルートウィジェットにレイアウトが無い | Designer で右クリック → レイアウトを適用 |
