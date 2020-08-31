def produce_name(member, variant):
    name = []
    if "l2ae" in member:
        return "L2-AE"
    if "aurora" in member:
        return "L2-AURORA"

    if "l1" in member:
        name.append("L1")
    elif "l0" in member or "smoothl1" in member.lower() or "huber" in member:
        name.append("L0")
    else:
        name.append("L2")

    if "beta0" not in member:
        name.append("KL")

    if "tsne" in member:
        name.append("TSNE")
    elif "sne" in member:
        name.append("SNE")

    if "nosampletrain" in member:
        name.append("NST")
    elif "nosample" in member:
        name.append("NS")

    if "fulllosstrue" in variant:
        name.append("LOG")
    else:
        name.append("EUC")

    return "-".join(name)