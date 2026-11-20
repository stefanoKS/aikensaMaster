# Inspection baseline audit (incomplete)

Baseline: `476689f623b74ca33345459ad996ebc02660b044` on `main`.
The local and remote `main` heads matched before the audit. No production
inspection or machine settings were changed.

This is a partial source and offscreen-geometry audit, not a production
qualification. Required model assignments, image processing, return shapes,
signals, counters, persistence, runtime classifications and actual machine
behavior have not yet been audited for every page.

## Legacy page inventory

The UI names below come from `UI_FILES`; registry entries come from
`aikensa/core/config/part_registry.yaml`. A dash means no registry entry, not
that the page is safe or unsupported. In particular, page 30 is an empty UI
but has a worker branch.

| Index | UI file | Registry part number |
|---:|---|---|
| 0 | mainPage.ui | - |
| 1 | calibration_cam1.ui | - |
| 2 | calibration_cam2.ui | - |
| 3 | camera_merge.ui | - |
| 4 | empty.ui | - |
| 5 | P82833W050P.ui | 82833W050P |
| 6 | P82832W040P.ui | 82832W040P |
| 7 | P82833W090P.ui | 82833W090P |
| 8 | P82832W080P.ui | 82832W080P |
| 9 | P82833W050PKENGEN.ui | 82833W050PKENGEN |
| 10 | P82832W040PKENGEN.ui | 82832W040PKENGEN |
| 11 | P82833W090PKENGEN.ui | 82833W090PKENGEN |
| 12 | P82832W080PKENGEN.ui | 82832W080PKENGEN |
| 13 | P82833W050PCLIPSOUNYUUKI.ui | 82833W050PCLIPSOUNYUUKI |
| 14 | P82832W040PCLIPSOUNYUUKI.ui | 82832W040PCLIPSOUNYUUKI |
| 15 | P82833W090PCLIPSOUNYUUKI.ui | 82833W090PCLIPSOUNYUUKI |
| 16 | P82832W080PCLIPSOUNYUUKI.ui | 82832W080PCLIPSOUNYUUKI |
| 17 | P808387UA1A.ui | 808387UA1A |
| 18 | P828447UA0A.ui | 828447UA0A |
| 19 | empty.ui | - |
| 20 | empty.ui | - |
| 21 | dailyTenken_new_01.ui | dailyTenken_01 |
| 22 | dailyTenken_new_02.ui | dailyTenken_02 |
| 23 | dailyTenken_new_03.ui | dailyTenken_03 |
| 24 | empty.ui | - |
| 25 | empty.ui | - |
| 26 | empty.ui | - |
| 27 | empty.ui | - |
| 28 | empty.ui | - |
| 29 | empty.ui | - |
| 30 | empty.ui | - |
| 31 | P658217UA0A.ui | 658217UA0A |
| 32 | P658207UA0A.ui | 658207UA0A |
| 33 | P658217UJ0A.ui | 658217UJ0A |
| 34 | P658207UJ0A.ui | 658207UJ0A |
| 35 | P658207LE0A.ui | P658207LE0A |
| 36 | P731957YA0A.ui | 731957YA0A |
| 37 | P808387YA0A.ui | 808387YA0A |
| 38 | P828387YA0A.ui | 828387YA0A |
| 39 | P828387YA1A.ui | 828387YA1A |
| 40 | P828397YA1A.ui | 828397YA1A |
| 41 | P828387YA6A.ui | 828387YA6A |
| 42 | P828397YA6A.ui | 828397YA6A |
| 43 | P658107YA0A.ui | 658107YA0A |
| 44 | P5902A510.ui | - |
| 45 | P5902A509.ui | - |
| 46 | P5819A107.ui | - |
| 47 | P8462284S00.ui | - |
| 48 | P828387YA6A_KATABU_NASHI.ui | 828387YA6A_KATABU_NASHI |
| 49 | P828397YA6A_KATABU_NASHI.ui | 828397YA6A_KATABU_NASHI |

The settings hub is appended after these 50 pages. The registry has 32
entries. The worker's `run()` has explicit product-specific branches for
9-18, 21-23, and 30-43 and 48-49. No such branch was found for 5-8 or 44-47.
These pages still participate in shared preview/routing logic; their actual
inspection capability needs runtime investigation. Do not infer it from the
test comment that claims 44-47 have legacy dispatch branches.

`execute_part_inspection()` has an exception-driven fallback to a different
inspector, but no call site was found in the application. This does not prove
that it is safe to activate. Its generic invocation does not match the real
inspector signatures.

## KENGEN routing

The menu's `P82832W080PKENGEN_button` displays `82832W080P` but routes to page
10. Page 10 and the registry identify `82832W040P`. Conversely the
`P82832W040PKENGEN_button` displays `82832W040P` but routes to page 12, whose
UI and registry identify `82832W080P`. The reversed routing is present in
the initial commit. Historical production records and the intended operator
workflow have not been verified; no mapping change was made.

## Display observations

In an offscreen Qt probe, the isolated settings controller reported a minimum
size hint of 1580 x 1179 logical pixels. A normal window requested at
1920 x 1080 was expanded to 1920 x 1179; requests at 1366 x 768 and
1280 x 720 were expanded to 1580 x 1179.

With the legacy pages loaded in an isolated stack, visible buttons extended
beyond the page viewport at 1366 x 768 and 1280 x 720. The main menu also had
four such buttons at 1920 x 1080. The probe did not test real pointer input,
camera frames or physical displays. On this offscreen backend calling `show()`
after `showFullScreen()` retained the fullscreen state; that does not establish
Windows behavior.

## Baseline checks and gaps

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen python -m pytest -q`
  from the repository root: 112 passed.
- `python -m compileall -q aikensa tests`: passed.
- No reference images were found under `tests/`; golden-image parity,
  performance, full application initialization, hardware and Windows display
  verification remain untested.
- The complete page-by-page model, geometry, signature, result, persistence
  and runtime audit remains outstanding. Do not treat this partial inventory
  as the G0 acceptance gate.
