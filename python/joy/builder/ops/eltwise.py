"""Element-wise binary operations for Joy dialect."""


def _broadcast_shape(s1, s2):
    r1, r2 = len(s1), len(s2)
    rank = max(r1, r2)
    result = []
    for i in range(rank):
        d1 = s1[i - (rank - r1)] if i >= rank - r1 else 1
        d2 = s2[i - (rank - r2)] if i >= rank - r2 else 1
        if d1 == d2:
            result.append(d1)
        elif d1 == 1 or d1 < 0:
            result.append(d2)
        elif d2 == 1 or d2 < 0:
            result.append(d1)
        else:
            result.append(max(d1, d2))
    return result


def add(lhs, rhs):
    result_shape = _broadcast_shape(lhs.shape, rhs.shape)
    return lhs.graph._create_op("joy.add", [lhs, rhs], result_shape, lhs.dtype)


def mul(lhs, rhs):
    result_shape = _broadcast_shape(lhs.shape, rhs.shape)
    return lhs.graph._create_op("joy.mul", [lhs, rhs], result_shape, lhs.dtype)
