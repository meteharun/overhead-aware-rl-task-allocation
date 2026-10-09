"""Build the platform grid. Usage: python scripts/generator/gen_platform.py

Twenty platforms, ordered by device count within each group, so the numbering
itself tells you roughly how hard a platform is.

**p1 to p15, training.** Two single device, eight two device, five three device.
Four matched pairs, each isolating link type on a different device pairing, so
"does the link matter" is a result rather than an anecdote:

    p3  vs p4     IMX + IMX,   10g vs usb
    p5  vs p6     IMX + QCS,   1g  vs 10g
    p7  vs p8     IMX + IQ,    10g vs usb
    p12 vs p13    IMX+IMX+QCS, both 10g vs one usb

**p16 to p20, never trained on.** Each of the first four changes exactly one
thing relative to a platform the policy did see, so a failure says which axis
broke:

    p16   IQ9100 alone            a device type only ever seen paired
    p17   IMX + IMX over 1g       a link type unseen on this pair
    p18   QCS6490 with IQ9100     a device pairing never linked directly
    p19   four IMX95              scale, one rung past p11
    p20   four devices, 3 links   everything at once, uncontrolled

The IMX95 scale ladder runs p1 (1 device), p3 (2), p11 (3), p19 (4, held out).
QCS6490 and IQ9100 are never directly linked in training, only as separate
leaves talking through the main device, which is what keeps p18 unseen.

Topology is always a star: device 0 is the main device and every link runs from
it. Two leaves have no direct link, so a message between them is relayed by the
main device and charged at every hop.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "data" / "platforms"

IMX, QCS, IQ = "NXP_IMX95", "QCOM_QCS6490", "QCOM_IQ9100"

# (device types, links as (main index, other index, link type), what it isolates)
# Ordered by device count, then by core count, with matched pairs adjacent.
PLATFORMS = [
    # --- one device ------------------------------------------------------
    ([IMX], [], "single device, six A55, homogeneous. The control."),
    ([QCS], [], "single device, mixed A55 and A78."),

    # --- two devices -----------------------------------------------------
    ([IMX, IMX], [(0, 1, "eth_10g")], "two devices over a cheap link."),
    ([IMX, IMX], [(0, 1, "usb")], "same capacity as p3, expensive link. The pair."),
    ([IMX, QCS], [(0, 1, "eth_1g")], "two CPU types over a medium link."),
    ([IMX, QCS], [(0, 1, "eth_10g")], "same capacity as p5, cheap link. The pair."),
    ([IMX, IQ], [(0, 1, "eth_10g")], "third CPU type over a cheap link."),
    ([IMX, IQ], [(0, 1, "usb")], "same capacity as p7, expensive link. The pair."),
    ([QCS, QCS], [(0, 1, "eth_1g")],
     "homogeneous pair that is not IMX, and QCS as the main device."),
    ([IQ, IQ], [(0, 1, "eth_10g")],
     "homogeneous IQ pair, and the only platform where IQ is the main device."),

    # --- three devices ---------------------------------------------------
    ([IMX, IMX, IMX], [(0, 1, "eth_10g"), (0, 2, "eth_10g")],
     "three identical devices. The middle rung of the IMX scale ladder."),
    ([IMX, IMX, QCS], [(0, 1, "eth_10g"), (0, 2, "eth_10g")],
     "three devices, both links cheap."),
    ([IMX, IMX, QCS], [(0, 1, "eth_10g"), (0, 2, "usb")],
     "same capacity as p12, one link expensive. The pair."),
    ([QCS, IMX, IMX], [(0, 1, "eth_1g"), (0, 2, "usb")],
     "QCS as the main device on a three device star, mixed links."),
    ([IMX, QCS, IQ], [(0, 1, "eth_1g"), (0, 2, "eth_1g")],
     "three devices, three CPU types. QCS and IQ are leaves, so they talk "
     "through the main device rather than directly."),

    # --- held out --------------------------------------------------------
    # each controlled against something the policy did see, so the
    # generalisation claim is per axis rather than one lump
    ([IQ], [],
     "the third CPU type on its own, which it never is in training. "
     "Held out: isolates the device."),
    ([IMX, IMX], [(0, 1, "eth_1g")],
     "p3 and p4 again over the one link type that pair never had. "
     "Held out: isolates the link."),
    ([QCS, IQ], [(0, 1, "eth_10g")],
     "two device types that are each well covered but never linked directly. "
     "Held out: isolates the device combination."),
    ([IMX, IMX, IMX, IMX], [(0, 1, "eth_10g"), (0, 2, "eth_10g"), (0, 3, "eth_10g")],
     "four identical devices, one link type, no CPU variety at all. "
     "Held out: isolates scale, one rung past p11."),
    ([IMX, IMX, QCS, IQ], [(0, 1, "eth_10g"), (0, 2, "eth_1g"), (0, 3, "usb")],
     "four devices, every link type at once. Held out: the hard case, and "
     "deliberately not controlled."),
]

# Never in the training split. p16 to p19 each change one thing relative to a
# training platform; p20 changes several and is the hard case.
EVAL_ONLY = (16, 17, 18, 19, 20)


def build(number, dev_types, links, note, catalogue):
    cores_of = {d["device_type"]: d["cores"] for d in catalogue["device_types"]}

    seen, device_ids = {}, []
    for t in dev_types:
        device_ids.append(f"{t}_{seen.get(t, 0)}")
        seen[t] = seen.get(t, 0) + 1

    cores = []
    for dev_id, t in zip(device_ids, dev_types):
        for c in cores_of[t]:
            cores.append({
                "core_id": f"{dev_id}_C{c['index']}",
                "device_id": dev_id,
                "cpu_type": c["cpu_type"],
                "core_class": c["core_class"],
            })

    wiring = ", ".join(f"{device_ids[b]} over {lt}" for _, b, lt in links) or "no links"
    return {
        "platform_id": f"platform_{number}",
        "note": f"{note} Star from {device_ids[0]}: {wiring}.",
        "devices": [{"device_id": d} for d in device_ids],
        "cores": cores,
        "links": [{"dev_a": device_ids[a], "dev_b": device_ids[b], "link_type": lt}
                  for a, b, lt in links],
    }


def main():
    # argparse so that --help prints the docstring instead of silently
    # rewriting all twenty platform files, which is what this does otherwise
    import argparse
    argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    catalogue = json.loads(
        (ROOT / "data" / "catalogues" / "hardware_catalogue.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("platform_*.json"):
        old.unlink()

    print(f"{'':<12}{'devices':>9}{'cores':>7}{'links':>7}{'pool':>8}   wiring")
    for i, (dev_types, links, note) in enumerate(PLATFORMS, start=1):
        plat = build(i, dev_types, links, note, catalogue)
        (OUT / f"{plat['platform_id']}.json").write_text(json.dumps(plat, indent=2) + "\n")
        lt = ", ".join(lk["link_type"] for lk in plat["links"]) or "-"
        print(f"{plat['platform_id']:<12}{len(plat['devices']):>9}"
              f"{len(plat['cores']):>7}{len(plat['links']):>7}"
              f"{'eval' if i in EVAL_ONLY else 'train':>8}   {lt}")

    print(f"\nheld out of training: {', '.join(f'platform_{i}' for i in EVAL_ONLY)}")


if __name__ == "__main__":
    main()
