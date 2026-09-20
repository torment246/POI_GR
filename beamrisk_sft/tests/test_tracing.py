from beamrisk_sft.tracing import LiveBeamStep, classify_beam_risk


GOLD = (10, 101, 102, 103, 104, 11, 2)
SID_POSITIONS = (1, 2, 3, 4)


def step(number: int, prefixes: list[tuple[int, ...]]) -> LiveBeamStep:
    assert len(prefixes) == 10
    return LiveBeamStep(
        step=number,
        prefixes=(tuple(prefixes),),
        scores=(tuple(float(-index) for index in range(10)),),
    )


def live_with_gold(length: int) -> LiveBeamStep:
    gold = GOLD[:length]
    others = [tuple([20 + index] * length) for index in range(9)]
    return step(length, [gold, *others])


def test_first_missing_sid_depth_uses_actual_tenth_boundary() -> None:
    steps = [live_with_gold(1), live_with_gold(2)]
    boundary = tuple([99] * 3)
    steps.append(step(3, [tuple([30 + i] * 3) for i in range(9)] + [boundary]))
    steps.extend([step(4, [tuple([40 + i] * 4) for i in range(10)]), step(5, [tuple([50 + i] * 5) for i in range(10)])])
    decision = classify_beam_risk(
        gold_token_ids=GOLD,
        sid_positions=SID_POSITIONS,
        live_steps=steps,
        sample_index=0,
        final_sequences=[tuple([70 + i] * 7) for i in range(10)],
    )
    assert decision.state == "miss@10"
    assert decision.risk_type == "first_prune"
    assert decision.first_prune_depth == 2
    assert decision.positive_token_ids == GOLD[:3]
    assert decision.negative_token_ids == boundary
    assert decision.boundary_rank == 10


def test_survive_then_rank_is_mutually_exclusive() -> None:
    steps = [live_with_gold(length) for length in range(1, 6)]
    wrong = (10, 201, 202, 203, 204, 11, 2)
    finals = [wrong, (10, 301, 302, 303, 304, 11, 2), GOLD]
    finals.extend(tuple([80 + i] * 7) for i in range(7))
    decision = classify_beam_risk(
        gold_token_ids=GOLD,
        sid_positions=SID_POSITIONS,
        live_steps=steps,
        sample_index=0,
        final_sequences=finals,
    )
    assert decision.state == "hit@10_not@1"
    assert decision.risk_type == "final_rank"
    assert decision.first_prune_depth is None
    assert decision.gold_rank == 3
    assert decision.negative_token_ids == wrong


def test_static_prefix_failure_does_not_become_sid_pair() -> None:
    steps = [step(1, [tuple([30 + i]) for i in range(10)])]
    steps.extend(step(length, [tuple([30 + i] * length) for i in range(10)]) for length in range(2, 6))
    decision = classify_beam_risk(
        gold_token_ids=GOLD,
        sid_positions=SID_POSITIONS,
        live_steps=steps,
        sample_index=0,
        final_sequences=[tuple([90 + i] * 7) for i in range(10)],
    )
    assert decision.state == "pre_sid_prune"
    assert decision.risk_type is None
