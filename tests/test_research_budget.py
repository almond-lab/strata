from dataclasses import replace
from test_research_review import make_project, question
from strata.research import ResearchPolicy, ResearchRound


def test_proposal_enumeration_stops_at_budget_and_keeps_remaining_questions_open(tmp_path):
    project=make_project(tmp_path)
    consumed=[]

    def unbounded_questions():
        i=0
        while True:
            consumed.append(i)
            yield replace(question(),change=f'construction {i}')
            i+=1

    round_=ResearchRound(project,project.task('and').base_design(),
                         ResearchPolicy(max_proposals=3,max_lines=1))
    plan=round_.plan(unbounded_questions())
    assert len(consumed)==4  # One lookahead detects truncation.
    assert plan['proposal_budget_reached']
    assert len(plan['selected'])==1 and len(plan['deferred'])==2
    assert not plan['skipped']
    assert not project.store.all('design')[1:]
