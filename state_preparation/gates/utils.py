class NotEnoughAuxQubits(Exception):
    """Exception raised when there are not enough auxiliary qubits available."""

    def __init__(self, required: int, available: int, num_controls: int):
        self.required = required
        self.available = available
        self.num_controls = num_controls
        super().__init__(
            f"Not enough auxiliary qubits. Required: {required}, Available: {available}, "
            f"for {num_controls} control qubits."
        )
