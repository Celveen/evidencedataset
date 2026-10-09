"""Evaluation holdout / seed selection: determinism and leakage rules."""

import pytest

from evidencetree.data.splits import SplitSpec, read_ids, select_splits, stable_rank, write_ids


def _infoseek_like(n_images=300, per_image=2):
    recs = []
    for i in range(n_images):
        stratum = "val_unseen_entity" if i % 2 else "val_unseen_question"
        for j in range(per_image):
            recs.append({
                "query_id": f"q{i}_{j}", "pool": "both",
                "metadata": {"image_id": f"img{i}", "data_split": stratum,
                             "entity_id": f"E{i % 50}"},
            })
    return recs


def _spec(**kw):
    base = dict(eval_n=40, seed_n=100, eval_strata="data_split", group_by="image_id",
                entity_field="entity_id", entity_disjoint_strata=("val_unseen_entity",),
                seed_backup=10)
    base.update(kw)
    return SplitSpec(**base)


def test_selection_is_deterministic_and_disjoint():
    recs = _infoseek_like()
    a = select_splits(recs, _spec())
    b = select_splits(list(reversed(recs)), _spec())
    assert a.eval_ids == b.eval_ids and a.seed_ids == b.seed_ids
    assert not set(a.eval_ids) & set(a.seed_ids + a.backup_seed_ids)
    assert len(a.seed_ids) == 100 and len(a.backup_seed_ids) == 10


def test_group_exclusivity_and_entity_disjointness():
    recs = _infoseek_like()
    res = select_splits(recs, _spec())
    by_id = {r["query_id"]: r["metadata"] for r in recs}
    eval_images = {by_id[q]["image_id"] for q in res.eval_ids}
    seed_images = {by_id[q]["image_id"] for q in res.seed_ids + res.backup_seed_ids}
    assert not eval_images & seed_images
    protected = {by_id[q]["entity_id"] for q in res.eval_ids
                 if by_id[q]["data_split"] == "val_unseen_entity"}
    assert protected
    assert not protected & {by_id[q]["entity_id"] for q in res.seed_ids}


def test_eval_strata_quotas_take_whole_groups():
    res = select_splits(_infoseek_like(), _spec())
    per = res.report["eval"]["per_stratum"]
    assert set(per) == {"val_unseen_entity", "val_unseen_question"}
    assert all(20 <= n <= 21 for n in per.values())  # quota 20, groups of 2


def test_pools_and_group_cap():
    recs = [{"query_id": f"t{i}", "pool": "seed", "image_id": f"img{i // 5}"} for i in range(100)]
    recs += [{"query_id": f"d{i}", "pool": "eval", "image_id": f"dev{i}"} for i in range(30)]
    res = select_splits(recs, SplitSpec(eval_n=10, seed_n=1000, group_by="image_id",
                                        max_seeds_per_group=1))
    assert all(q.startswith("d") for q in res.eval_ids) and len(res.eval_ids) == 10
    assert len(res.seed_ids) == 20  # one per image
    assert res.report["seed_shortfall"] == 980


def test_bad_records_raise():
    with pytest.raises(ValueError, match="duplicate"):
        select_splits([{"query_id": "a", "pool": "seed"}] * 2, SplitSpec())
    with pytest.raises(ValueError, match="pool"):
        select_splits([{"query_id": "a", "pool": "train"}], SplitSpec())


def test_stable_rank_and_id_files(tmp_path):
    assert stable_rank("x", "s") == stable_rank("x", "s") != stable_rank("x", "t")
    write_ids(tmp_path / "ids.txt", ["b", "a"])
    assert read_ids(tmp_path / "ids.txt") == ["b", "a"]
