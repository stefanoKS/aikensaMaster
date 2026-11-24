# Inspection migration status

Use the [legacy page inventory](../settings/inspection_baseline_audit.md#legacy-page-inventory)
for names and their existing widget indices. A page in the inventory is not
evidence of a supported or qualified inspection.

| Widget indices | Profile identity | Designer page | Menu route | Algorithm / execution mode | Software workflow | Machine qualification |
|---|---|---|---|---|---|---|
| 5–8 | `82833W050P`, `82832W040P`, `82833W090P`, `82832W080P` | Populated | Disabled in `c8d4868` | Shared legacy preview/routing; no dedicated dispatch found | Menu policy tested; inspection NOT TESTED | NOT TESTED |
| 9–16, 18, 31–43, 48–49 | See inventory | Populated | Enabled | Legacy production branches; widget 36 also has a separate daily mode | NOT TESTED per profile | NOT TESTED |
| 17 | `808387UA1A` | Populated | Enabled | Coordinator strategy; isolated parity tests | NOT TESTED end-to-end | NOT TESTED |
| 21–23 | `dailyTenken_01`–`dailyTenken_03` | Populated | Via daily inspection | Shared daily strategy with approved per-position revisions | PASS for synthetic Settings-to-worker, SQLite and image workflow; hardware and MySQL NOT TESTED | NOT TESTED |
| 30 | Historical `658217UA0A` alias (not an independent registered profile) | Empty | Disabled; current menu selects 31 | Historical production worker branch with its own geometry and inference | NOT TESTED; do not enable | NOT TESTED |
| 44–47 | See UI inventory; no registry entry | Populated | Enabled | Shared legacy preview/routing; no dedicated dispatch found | NOT TESTED | NOT TESTED |
| 4, 19, 20, 24–29 | None | Empty | Disabled | Reserved; no inspector found | Not applicable | Not applicable |
| Future profiles | Not registered | Not implemented | Disabled | Dynamic application registration not implemented | BLOCKED | NOT TESTED |
