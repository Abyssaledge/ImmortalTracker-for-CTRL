from ipdb import set_trace

def merge_bidirection(trks1, trks2, config):
    num_all = len(trks1) + len(trks2)
    birthdays1 = [t.birthday for t in trks1]
    birthdays2 = [t.birthday for t in trks2]
    birthdays2_minous_1 = [i - 1 for i in birthdays2]
    mutual_birthday = list(set(birthdays1).intersection(set(birthdays2_minous_1)))
    assert len(mutual_birthday) == 1

    for t in trks1:
        t.freeze()
    for t in trks2:
        t.freeze()

    m1 = {ts:i for i, ts in enumerate(birthdays1)}
    m2 = {ts:i for i, ts in enumerate(birthdays2)}

    set_trace()

    extended_trks = []
    for ts in inter_ts:
        i1, i2 = m1[ts], m2[ts+1]
        t1, t2 = trks1[i1], trks2[i2]
        t1.extend(t2)
        extended_trks.append(t1)
        trks1.pop(i1)
        trks2.pop(i2)
    
    assert len(extended_trks) * 2 + len(trks1) + len(trks2) == num_all
    merged_trks = extended_trks + trks1 + trks2

    all_time_stamps = set()
    for trk in merged_trks:
        all_time_stamps = all_time_stamps.union(set(trk.time_stamp_history))
    all_time_stamps = sorted(list(all_time_stamps))

    # keep the origin format
    ID_out, box_out, state_out, type_out = list(), list(), list(), list()
    for ts in all_time_stamps:
        ids_, bboxes_, states_, types_ = list(), list(), list(), list()
        for trk_id, trk in enumerate(merged_trks):
            r = trk.get_a_frame_result(ts)
            if r is None:
                continue
            ids_.append(trk_id)
            bboxes_.append(r['bboxes'])
            states_.append(r['state'])
            types_.append(r['type'])
        ID_out.append(ids_)
        box_out.append(bboxes_)
        state_out.append(states_)
        type_out.append(types_)
    return ID_out, box_out, state_out, type_out
    
        



