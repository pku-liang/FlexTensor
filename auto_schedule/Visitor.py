

class BaseVisitor(object):
    def __init__(self):
        class VisitMessage(object):
            def __init__(self, env):
                self.env = env
                self.visited = set()
                self.track = []

        self.VM = VisitMessage

    def get_visit_msg(self, env):
        raise NotImplementedError("BaseVisitor cannot call visit")

    def name(self):
        return self.__class__.__name__


class ReverseBFSVisitor(BaseVisitor):
    def __init__(self):
        super(ReverseBFSVisitor, self).__init__()

    def get_visit_msg(self, env):
        vm = self.VM(env)
        for op in env.end:
            self._bfs(op, vm)
        return vm

    def _bfs(self, op, vm):
        if not op or op in vm.visited:
            return
        vm.track.append(op)
        vm.visited.add(op)
        for t in op.input_tensors:
            full_visited = True
            # If an operation has some child operation not visited, do not visit it
            if t.op in vm.env.down_graph:
                for c_op in vm.env.down_graph[t.op]:
                    if c_op not in vm.visited:
                        full_visited = False
                        break
            if full_visited:
                self._bfs(t.op, vm)


VISITOR_TABLE = {}


def register_visitor(visitor_class):
    VISITOR_TABLE[visitor_class.__name__] = visitor_class


register_visitor(ReverseBFSVisitor)
