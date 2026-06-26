from lwmf.schema import Trajectory, Turn
from lwmf.data.format import trajectory_to_messages, expand_to_turns

TRAJ = Trajectory(
    scenario="terminal", system="SYS",
    turns=[Turn("ls", "a.txt"), Turn("cat a.txt", "hello")],
)

def test_messages_shape():
    m = trajectory_to_messages(TRAJ)
    assert m[0] == {"role": "system", "content": "SYS"}
    assert m[1] == {"role": "user", "content": "ls"}
    assert m[2] == {"role": "assistant", "content": "a.txt"}
    assert m[3] == {"role": "user", "content": "cat a.txt"}
    assert m[4] == {"role": "assistant", "content": "hello"}

def test_expand_to_turns():
    ex = expand_to_turns(TRAJ)
    assert len(ex) == 2
    # turn 0: only system + first user action, target = first observation
    assert ex[0].prefix_messages == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "ls"},
    ]
    assert ex[0].target == "a.txt"
    # turn 1: includes the resolved first turn (both roles) + second action
    assert ex[1].prefix_messages[-1] == {"role": "user", "content": "cat a.txt"}
    assert ex[1].prefix_messages[2] == {"role": "assistant", "content": "a.txt"}
    assert ex[1].target == "hello"
