"""
编排层状态机

该模块定义了StateMachine类，为每一个agent策略提供状态机支持。
"""

from typing import Dict, List, Optional


class StateMachine:
    """
    StateMachine类 - 编排层状态机

    为每一个agent策略提供状态机支持，实现基于有限状态机(FSM)的业务逻辑控制。

    职责：
        - 管理业务流程的状态转换
        - 验证状态转换的合法性
        - 提供状态可达性查询

    使用示例：
        >>> state_map = {
        ...     "INIT": ["ANALYZING"],
        ...     "ANALYZING": ["RETRIEVING", "VALIDATING"],
        ...     "RETRIEVING": ["GENERATING"],
        ...     "VALIDATING": ["GENERATING"],
        ...     "GENERATING": ["COMPLETED"],
        ...     "COMPLETED": []
        ... }
        >>> sm = StateMachine("session_001", state_map)
        >>> sm.validate_state("INIT", "ANALYZING")  # True
        >>> sm.transition("INIT", "ANALYZING")  # "ANALYZING"

    Attributes:
        _session_id: 任务id
        _state_transition_map: 状态转变表，key为当前状态，value为可转换到的状态列表
    """

    def __init__(
        self,
        session_id: str,
        state_transition_map: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """
        初始化状态机

        Args:
            session_id: 任务id
            state_transition_map: 状态转变表，key为当前状态，value为可转换到的状态列表。
                                如果为None，则使用空字典。

        Raises:
            ValueError: session_id为空时抛出
        """
        if not session_id:
            raise ValueError("session_id不能为空")

        self._session_id: str = session_id
        self._state_transition_map: Dict[str, List[str]] = state_transition_map or {}

    @property
    def session_id(self) -> str:
        """
        获取任务id（只读属性）

        Returns:
            str: 任务id
        """
        return self._session_id

    @property
    def state_transition_map(self) -> Dict[str, List[str]]:
        """
        获取状态转变表（只读属性）

        Returns:
            Dict[str, List[str]]: 状态转变表
        """
        return self._state_transition_map.copy()

    def transition(self, current_state: str, target_state: str) -> str:
        """
        变更当前状态到目标状态

        验证状态转换的合法性，如果合法则返回目标状态，否则抛出异常。

        Args:
            current_state: 当前状态
            target_state: 目标状态

        Returns:
            str: 目标状态

        Raises:
            ValueError: 参数为空或状态转换不合法时抛出

        Example:
            >>> sm.transition("INIT", "ANALYZING")
            "ANALYZING"
        """
        if not current_state:
            raise ValueError("current_state不能为空")
        if not target_state:
            raise ValueError("target_state不能为空")

        # 验证状态转换的合法性
        if not self.validate_state(current_state, target_state):
            raise ValueError(
                f"状态转换不合法：无法从 '{current_state}' 转换到 '{target_state}'。"
                f"当前状态可转换到的状态：{self.get_reachable_state(current_state)}"
            )

        return target_state

    def validate_state(self, current_state: str, target_state: str) -> bool:
        """
        验证当前状态是否能到目标状态

        检查状态转换表中是否存在从current_state到target_state的转换路径。

        Args:
            current_state: 当前状态
            target_state: 目标状态

        Returns:
            bool: 是否可以转换，True表示可以转换，False表示不可以转换

        Example:
            >>> sm.validate_state("INIT", "ANALYZING")
            True
            >>> sm.validate_state("INIT", "COMPLETED")
            False
        """
        if not current_state or not target_state:
            return False

        # 获取当前状态可到达的状态列表
        reachable_states = self.get_reachable_state(current_state)

        # 检查目标状态是否在可到达状态列表中
        return target_state in reachable_states

    def get_reachable_state(self, current_state: str) -> List[str]:
        """
        返回从当前状态可达到的状态列表

        从状态转换表中获取当前状态可以直接转换到的所有状态。

        Args:
            current_state: 当前状态

        Returns:
            List[str]: 可达到的状态列表，如果当前状态不在转换表中，返回空列表

        Example:
            >>> sm.get_reachable_state("INIT")
            ["ANALYZING"]
            >>> sm.get_reachable_state("UNKNOWN_STATE")
            []
        """
        if not current_state:
            return []

        # 从状态转换表中获取当前状态可到达的状态列表
        return self._state_transition_map.get(current_state, []).copy()

    def add_state_transition(self, from_state: str, to_states: List[str]) -> None:
        """
        添加状态转换规则

        向状态转换表中添加新的状态转换规则。

        Args:
            from_state: 起始状态
            to_states: 可转换到的状态列表

        Raises:
            ValueError: 参数为空时抛出

        Example:
            >>> sm.add_state_transition("INIT", ["ANALYZING", "LOADING"])
        """
        if not from_state:
            raise ValueError("from_state不能为空")
        if not to_states:
            raise ValueError("to_states不能为空")

        # 如果起始状态已存在，合并状态列表
        if from_state in self._state_transition_map:
            existing_states = set(self._state_transition_map[from_state])
            new_states = set(to_states)
            self._state_transition_map[from_state] = list(existing_states | new_states)
        else:
            self._state_transition_map[from_state] = to_states.copy()

    def remove_state(self, state: str) -> None:
        """
        移除状态及其所有转换规则

        从状态转换表中移除指定状态及其所有转换规则。

        Args:
            state: 要移除的状态

        Example:
            >>> sm.remove_state("ANALYZING")
        """
        if not state:
            return

        # 移除该状态作为起始状态的所有转换规则
        if state in self._state_transition_map:
            del self._state_transition_map[state]

        # 移除其他状态中指向该状态的转换
        for from_state in self._state_transition_map:
            if state in self._state_transition_map[from_state]:
                self._state_transition_map[from_state].remove(state)

    def has_state(self, state: str) -> bool:
        """
        检查状态是否存在

        检查指定状态是否在状态转换表中（作为起始状态或目标状态）。

        Args:
            state: 要检查的状态

        Returns:
            bool: 状态是否存在

        Example:
            >>> sm.has_state("INIT")
            True
        """
        if not state:
            return False

        # 检查是否作为起始状态存在
        if state in self._state_transition_map:
            return True

        # 检查是否作为目标状态存在
        for to_states in self._state_transition_map.values():
            if state in to_states:
                return True

        return False

    def get_all_states(self) -> List[str]:
        """
        获取所有状态列表

        返回状态转换表中的所有状态（包括起始状态和目标状态）。

        Returns:
            List[str]: 所有状态的列表

        Example:
            >>> sm.get_all_states()
            ["INIT", "ANALYZING", "RETRIEVING", "GENERATING", "COMPLETED"]
        """
        states = set()

        # 添加所有起始状态
        states.update(self._state_transition_map.keys())

        # 添加所有目标状态
        for to_states in self._state_transition_map.values():
            states.update(to_states)

        return list(states)

    def clear(self) -> None:
        """
        清空状态转换表

        清空所有状态转换规则。
        """
        self._state_transition_map.clear()

    def __repr__(self) -> str:
        """返回状态机的字符串表示"""
        return (
            f"StateMachine("
            f"session_id='{self._session_id}', "
            f"state_count={len(self._state_transition_map)})"
        )
