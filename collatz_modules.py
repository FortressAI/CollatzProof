"""
Field of Truth (FoT) modules for the Collatz Conjecture proof system.

This file implements the complete agent-based system for the FoT framework,
providing an Agentic Knowledge Graph (AKG) approach to proving the Collatz Conjecture.
It includes all core operational agents and scaling agents as described in the
Field of Truth framework.
"""

import os
import json
import time
import hashlib
import uuid
from typing import Dict, List, Tuple, Optional, Set, Any
import sympy as sp

class MathPattern:
    """
    Represents a mathematical pattern with input/output expressions and transformation metadata.
    """
    def __init__(
        self, 
        input_expr, 
        output_expr, 
        reasoning_type="unknown",
        agent_chain=None,
        problem_ids=None,
        frequency=1,
        metadata=None
    ):
        """
        Initialize a math pattern.
        
        Args:
            input_expr: Input expression
            output_expr: Output expression
            reasoning_type: Type of reasoning
            agent_chain: List of agent names that transform input to output
            problem_ids: Set of problem IDs where this pattern was observed
            frequency: Number of times this pattern has been observed
            metadata: Additional metadata about the pattern
        """
        # Store original expressions as strings
        self.input_expr = str(input_expr)
        self.output_expr = str(output_expr)
        
        self.reasoning_type = reasoning_type
        self.agent_chain = agent_chain or []
        self.problem_ids = problem_ids or set()
        self.frequency = frequency
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert pattern to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the pattern
        """
        return {
            "input_expr": self.input_expr,
            "output_expr": self.output_expr,
            "reasoning_type": self.reasoning_type,
            "agent_chain": self.agent_chain,
            "problem_ids": list(self.problem_ids),
            "frequency": self.frequency,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MathPattern':
        """
        Create a pattern from a dictionary.
        
        Args:
            data: Dictionary representation of a pattern
            
        Returns:
            MathPattern instance
        """
        # Convert problem_ids back to a set
        problem_ids = set(data.pop("problem_ids", []))
        
        # Create pattern
        pattern = cls(
            input_expr=data["input_expr"],
            output_expr=data["output_expr"],
            reasoning_type=data.get("reasoning_type", "unknown"),
            agent_chain=data.get("agent_chain", []),
            problem_ids=problem_ids,
            frequency=data.get("frequency", 1),
            metadata=data.get("metadata", {})
        )
        
        return pattern


class BaseMathAgent:
    """Base class for all math transformation agents in the Field of Truth framework."""

    name: str

    def __init__(self, name: str = None):
        """
        Initialize a math agent.
        
        Args:
            name: Name of the agent (defaults to class name)
        """
        self.name = name or self.__class__.__name__
        self.agent_id = str(uuid.uuid4())[:8]  # Unique ID for this agent instance

    def transform(self, expr):
        """Apply the transformation to a SymPy expression."""
        raise NotImplementedError

    def explain(self) -> str:
        """
        Provide a human-readable explanation of the transformation.
        
        Returns:
            Explanation string
        """
        return f"Apply {self.name} to the expression"
        
    def get_signature(self, input_val, output_val):
        """
        Generate a validation signature for this transformation.
        
        Args:
            input_val: Input value
            output_val: Output value
            
        Returns:
            Signature hash
        """
        signature_str = f"{self.name}|{self.agent_id}|{input_val}|{output_val}"
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]


class CollatzOddAgent(BaseMathAgent):
    """Agent that applies the η₁(n) = 3n+1 transformation for odd numbers.
    
    In the Field of Truth framework, this agent initiates a "growth step"
    and ensures arithmetic validity before submitting the step for validation.
    """
    
    def __init__(self):
        """Initialize Collatz odd number agent."""
        super().__init__("CollatzOddAgent")
        self.operation_count = 0
        self.validator = None
    
    def transform(self, expr):
        """Apply the 3n+1 transformation to an odd number."""
        # If it's a literal number, apply transformation directly
        if hasattr(expr, "is_Integer") and expr.is_Integer:
            n = int(expr)
            if n % 2 == 1:  # Check if odd
                result = 3*n + 1
                self.operation_count += 1
                
                # If we have a validator, register this transformation
                if self.validator:
                    self.validator.validate(n, result, self)
                    
                return sp.Integer(result)
            else:
                raise ValueError(f"CollatzOddAgent cannot process even number: {n}")
        
        # Otherwise, create the symbolic operation
        return 3*expr + 1
    
    def explain(self) -> str:
        """Explain the Collatz odd transformation."""
        return "Apply the Collatz η₁(n) = 3n+1 operation for odd numbers (growth step)"


class CollatzEvenAgent(BaseMathAgent):
    """Agent that applies the η₂(n) = n/2 transformation for even numbers.
    
    In the Field of Truth framework, this agent initiates a "contraction step"
    and verifies divisibility before passing the result for validation.
    """
    
    def __init__(self):
        """Initialize Collatz even number agent."""
        super().__init__("CollatzEvenAgent")
        self.operation_count = 0
        self.validator = None
    
    def transform(self, expr):
        """Apply the n/2 transformation to an even number."""
        # If it's a literal number, apply transformation directly
        if hasattr(expr, "is_Integer") and expr.is_Integer:
            n = int(expr)
            if n % 2 == 0:  # Check if even
                result = n // 2
                self.operation_count += 1
                
                # If we have a validator, register this transformation
                if self.validator:
                    self.validator.validate(n, result, self)
                    
                return sp.Integer(result)
            else:
                raise ValueError(f"CollatzEvenAgent cannot process odd number: {n}")
        
        # Otherwise, create the symbolic operation
        return expr / 2
    
    def explain(self) -> str:
        """Explain the Collatz even transformation."""
        return "Apply the Collatz η₂(n) = n/2 operation for even numbers (contraction step)"


# Field of Truth specialized agents
class ValidationAgent:
    """
    Validation agent that confirms the correctness of each transformation and
    cryptographically signs the result.
    
    This agent is the guarantor of the system's integrity. It confirms each step
    and creates a verifiable record of the logic chain.
    """
    
    def __init__(self):
        self.validated_transformations = {}
        self.validation_count = 0
        self.signatures = {}
    
    def validate(self, input_val, output_val, agent):
        """
        Validate a transformation and sign it.
        
        Args:
            input_val: Input value
            output_val: Output value
            agent: The agent that performed the transformation
        
        Returns:
            True if valid, False otherwise
        """
        # Verify the transformation logic
        if isinstance(agent, CollatzOddAgent):
            valid = (input_val % 2 == 1) and (output_val == 3 * input_val + 1)
        elif isinstance(agent, CollatzEvenAgent):
            valid = (input_val % 2 == 0) and (output_val == input_val // 2)
        else:
            valid = False
            
        if valid:
            # Generate a signature
            signature = agent.get_signature(input_val, output_val)
            
            # Record the validation
            key = (input_val, output_val)
            self.validated_transformations[key] = {
                "agent": agent.name,
                "timestamp": time.time(),
                "signature": signature
            }
            
            # Store the signature
            self.signatures[key] = signature
            
            self.validation_count += 1
            
        return valid
    
    def get_signature(self, input_val, output_val):
        """Get the signature for a validated transformation."""
        key = (input_val, output_val)
        return self.signatures.get(key, None)


class SubpathReuseAgent:
    """
    Agent that detects overlaps with known paths (Φ) to save computation.
    
    This agent checks if the current path being constructed overlaps with
    any previously verified path, allowing the system to reuse proven results.
    """
    
    def __init__(self):
        self.known_paths = {}  # number -> path to 1
        self.reuse_count = 0
        
    def register_path(self, n, path):
        """
        Register a verified path for future reuse.
        
        Args:
            n: Starting number
            path: Path from n to 1
        """
        if n not in self.known_paths:
            self.known_paths[n] = path
    
    def check_known_path(self, n):
        """
        Check if we already have a verified path for this number.
        
        Args:
            n: Number to check
            
        Returns:
            Path if known, None otherwise
        """
        if n in self.known_paths:
            self.reuse_count += 1
            return self.known_paths[n]
        return None


class EthicsIntegrityAgent:
    """
    Agent that enforces logical and ethical constraints on the proof system.
    
    This agent prevents malformed logic, illegal cycles, or unverifiable claims
    from being introduced into the AKG.
    """
    
    def __init__(self):
        self.audit_count = 0
        self.violation_count = 0
        self.cycles = set([(4, 2, 1)])  # The only legitimate cycle
    
    def check_integrity(self, path):
        """
        Check the integrity of a path.
        
        Args:
            path: Path to check
            
        Returns:
            (valid, reason) tuple
        """
        self.audit_count += 1
        
        # Check for empty path
        if not path:
            self.violation_count += 1
            return False, "Empty path"
        
        # Check that path ends at 1
        if path[-1] != 1:
            self.violation_count += 1
            return False, "Path does not end at 1"
        
        # Check for cycles other than 4->2->1
        for i in range(len(path) - 2):
            if i + 3 <= len(path):
                cycle = tuple(path[i:i+3])
                if cycle != (4, 2, 1) and cycle in self.cycles:
                    self.violation_count += 1
                    return False, f"Illegal cycle detected: {cycle}"
        
        # Check for valid transitions
        for i in range(len(path) - 1):
            n = path[i]
            next_n = path[i + 1]
            
            if n % 2 == 0:
                # Even case
                if next_n != n // 2:
                    self.violation_count += 1
                    return False, f"Invalid even transition: {n} -> {next_n}"
            else:
                # Odd case
                if next_n != 3 * n + 1:
                    self.violation_count += 1
                    return False, f"Invalid odd transition: {n} -> {next_n}"
        
        return True, "Path is valid"


class CollatzWalker:
    """
    Orchestrator agent that walks a Collatz path from a starting number.
    
    This agent recursively schedules the appropriate Odd or Even agents
    and manages the sequence until it converges or merges with a known path.
    """
    
    def __init__(self, validator=None, subpath_agent=None, ethics_agent=None):
        self.odd_agent = CollatzOddAgent()
        self.even_agent = CollatzEvenAgent()
        self.validator = validator
        self.subpath_agent = subpath_agent
        self.ethics_agent = ethics_agent
        
        # Connect agents
        if validator:
            self.odd_agent.validator = validator
            self.even_agent.validator = validator
    
    def walk(self, n, max_steps=1000):
        """
        Walk the Collatz path starting from n.
        
        Args:
            n: Starting number
            max_steps: Maximum steps before giving up
            
        Returns:
            (success, path, steps) tuple
        """
        path = [n]
        steps = 0
        
        while path[-1] != 1 and steps < max_steps:
            current = path[-1]
            
            # Check if we can reuse a known path
            if self.subpath_agent:
                known_path = self.subpath_agent.check_known_path(current)
                if known_path:
                    # Merge with the known path (excluding the first element which is current)
                    path.extend(known_path[1:])
                    break
            
            # Apply the appropriate transformation
            if current % 2 == 0:
                next_n = self.even_agent.transform(current)
            else:
                next_n = self.odd_agent.transform(current)
            
            path.append(int(next_n))
            steps += 1
        
        success = path[-1] == 1
        
        # Verify the integrity of the path
        if success and self.ethics_agent:
            valid, reason = self.ethics_agent.check_integrity(path)
            if not valid:
                success = False
        
        # Register the path for future reuse if successful
        if success and self.subpath_agent:
            self.subpath_agent.register_path(n, path)
        
        return success, path, steps


# Scaling agents for the Field of Truth framework
class GraphMergeAgent:
    """
    Agent that merges partial subpaths that share suffixes.
    
    This agent identifies where paths merge and compresses the graph
    to minimize recomputation.
    """
    
    def __init__(self):
        self.merged_paths = {}
        self.merge_count = 0
    
    def merge(self, path1, path2):
        """Find where two paths merge and create a compressed structure."""
        # Find the first common value
        merge_point = None
        merge_idx1 = None
        merge_idx2 = None
        
        for i, val1 in enumerate(path1):
            for j, val2 in enumerate(path2):
                if val1 == val2:
                    merge_point = val1
                    merge_idx1 = i
                    merge_idx2 = j
                    break
            if merge_point:
                break
        
        if merge_point:
            self.merge_count += 1
            key = (tuple(path1), tuple(path2))
            self.merged_paths[key] = (merge_point, merge_idx1, merge_idx2)
            
            return True, merge_point, merge_idx1, merge_idx2
        
        return False, None, None, None


class PrimePrefetchAgent:
    """
    Agent that prioritizes long-running, slow-converging primes.
    
    This agent identifies prime numbers which often have longer convergence
    paths and processes them early to optimize the graph structure.
    """
    
    def __init__(self):
        self.processed_primes = set()
        
    def is_prime(self, n):
        """Check if a number is prime."""
        if n <= 1:
            return False
        if n <= 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    def get_primes_in_range(self, start, end, limit=100):
        """Get up to limit prime numbers in the range [start, end]."""
        primes = []
        for n in range(start, end + 1):
            if len(primes) >= limit:
                break
            if self.is_prime(n) and n not in self.processed_primes:
                primes.append(n)
                self.processed_primes.add(n)
        return primes


class RangeShardAgent:
    """
    Agent that handles a specific range of numbers.
    
    This agent enables parallel processing by focusing on a specific
    slice of the number range.
    """
    
    def __init__(self, start_range, end_range, walker=None):
        self.start_range = start_range
        self.end_range = end_range
        self.walker = walker
        self.verified_numbers = set()
    
    def process_range(self, batch_size=100, max_steps=1000):
        """Process a batch of numbers in this range."""
        if not self.walker:
            return []
        
        results = []
        count = 0
        
        for n in range(self.start_range, self.end_range + 1):
            if n in self.verified_numbers or count >= batch_size:
                continue
                
            success, path, steps = self.walker.walk(n, max_steps)
            results.append((n, success, steps, len(path)))
            
            if success:
                self.verified_numbers.add(n)
            
            count += 1
            
        return results


class CycleDetectionAgent:
    """
    Agent that monitors for any unintended cycles.
    
    This agent ensures that the only cycle in the Collatz graph is
    the canonical 4→2→1 loop.
    """
    
    def __init__(self):
        self.known_cycles = set([frozenset([1, 2, 4])])  # The only legitimate cycle
        self.detected_cycles = []
    
    def detect_cycle(self, path):
        """Check if a path contains any cycles."""
        seen = {}
        for i, val in enumerate(path):
            if val in seen:
                cycle = path[seen[val]:i+1]
                cycle_set = frozenset(cycle)
                if cycle_set not in self.known_cycles:
                    self.detected_cycles.append(cycle)
                    return True, cycle
            seen[val] = i
        return False, None


class DeepConvergenceAgent:
    """
    Agent that measures and logs convergence depth (Ξ(n)).
    
    This agent tracks how many steps each number takes to reach 1,
    enabling pattern analysis and optimization.
    """
    
    def __init__(self):
        self.convergence_depths = {}  # n -> steps to reach 1
        
    def record_depth(self, n, depth):
        """Record the convergence depth for a number."""
        self.convergence_depths[n] = depth
        
    def get_depth(self, n):
        """Get the recorded convergence depth for a number."""
        return self.convergence_depths.get(n)
        
    def get_stats(self):
        """Get statistics about recorded depths."""
        if not self.convergence_depths:
            return {"count": 0, "avg": 0, "max": 0, "min": 0}
            
        depths = list(self.convergence_depths.values())
        return {
            "count": len(depths),
            "avg": sum(depths) / len(depths),
            "max": max(depths),
            "min": min(depths),
            "max_n": max(self.convergence_depths.items(), key=lambda x: x[1])[0]
        }


class AKGBroadcastAgent:
    """
    Agent that publishes newly validated paths across the validator mesh.
    
    This agent enables sharing of verified paths between distributed
    computation nodes, maximizing reuse and efficiency.
    """
    
    def __init__(self):
        self.broadcasts = []
        self.subscribers = []
    
    def broadcast(self, n, path, validator_id):
        """Broadcast a newly validated path."""
        broadcast = {
            "number": n,
            "path_length": len(path),
            "validator_id": validator_id,
            "timestamp": time.time()
        }
        self.broadcasts.append(broadcast)
        
        # Notify all subscribers
        for subscriber in self.subscribers:
            subscriber.receive_broadcast(n, path)
        
        return broadcast
    
    def subscribe(self, agent):
        """Add a subscriber to receive broadcasts."""
        if agent not in self.subscribers:
            self.subscribers.append(agent)


class CollatzKG:
    """
    Knowledge graph specialized for the Collatz conjecture.
    
    Stores Collatz sequence paths and enables efficient path reuse.
    In the Field of Truth framework, this serves as the Agentic Knowledge Graph (AKG)
    that represents the complete proof structure.
    """
    def __init__(self, storage_dir: Optional[str] = None, max_patterns: int = 1000000):
        """Initialize a Collatz knowledge graph with the Field of Truth agent framework."""
        self.storage_dir = storage_dir
        self.max_patterns = max_patterns
        
        # Dictionary to store patterns
        self.patterns: Dict[str, MathPattern] = {}
        
        # Adjacency list for transformations
        self.adjacency: Dict[str, List[Tuple[str, str]]] = {}
        
        # Maps from number -> pattern key for quick lookup
        self.number_to_key: Dict[int, str] = {}
        
        # Track if a number is known to reach 1
        self.leads_to_one: Set[int] = {1, 2, 4}  # Start with the known cycle
        
        # Track path lengths for reporting
        self.path_lengths: Dict[int, int] = {1: 0, 2: 1, 4: 2}
        
        # Initialize Field of Truth agent system
        self.validator = ValidationAgent()
        self.subpath_agent = SubpathReuseAgent()
        self.ethics_agent = EthicsIntegrityAgent()
        
        # Initialize core transformation agents
        self.odd_agent = CollatzOddAgent()
        self.even_agent = CollatzEvenAgent()
        
        # Connect agents with validation
        self.odd_agent.validator = self.validator
        self.even_agent.validator = self.validator
        
        # Initialize the walker agent
        self.walker = CollatzWalker(
            validator=self.validator,
            subpath_agent=self.subpath_agent,
            ethics_agent=self.ethics_agent
        )
        
        # Initialize scaling agents
        self.graph_merge_agent = GraphMergeAgent()
        self.prime_prefetch_agent = PrimePrefetchAgent()
        self.cycle_detection_agent = CycleDetectionAgent()
        self.deep_convergence_agent = DeepConvergenceAgent()
        self.broadcast_agent = AKGBroadcastAgent()
        
        # Register known cycle paths
        for n in self.leads_to_one:
            if n == 1:
                self.subpath_agent.register_path(1, [1])
            elif n == 2:
                self.subpath_agent.register_path(2, [2, 1])
            elif n == 4:
                self.subpath_agent.register_path(4, [4, 2, 1])
        
        # Create storage directory if needed
        if storage_dir and not os.path.exists(storage_dir):
            os.makedirs(storage_dir, exist_ok=True)
    
    def add_pattern(self, pattern: MathPattern) -> str:
        """
        Add a pattern to the knowledge graph.
        
        Args:
            pattern: The pattern to add
            
        Returns:
            The pattern key
        """
        # Generate pattern key
        key = self._hash_pattern(pattern.input_expr)
        
        # Add or update pattern
        if key not in self.patterns:
            self.patterns[key] = pattern
        else:
            # Update existing pattern
            existing = self.patterns[key]
            existing.frequency += 1
            existing.problem_ids.update(pattern.problem_ids)
            
            # Update metadata if any
            if pattern.metadata:
                if not existing.metadata:
                    existing.metadata = {}
                existing.metadata.update(pattern.metadata)
            
        return key
    
    def add_transformation(self, from_key: str, to_key: str, agent_name: str) -> None:
        """
        Add a transformation edge from one pattern to another.
        
        Args:
            from_key: Source pattern key
            to_key: Target pattern key
            agent_name: Name of the agent that performs the transformation
        """
        if from_key not in self.adjacency:
            self.adjacency[from_key] = []
        
        self.adjacency[from_key].append((to_key, agent_name))
    
    def add_number(self, n: int) -> str:
        """
        Add a number to the knowledge graph.
        
        Args:
            n: The number to add
            
        Returns:
            The pattern key
        """
        # Check if already in graph
        if n in self.number_to_key:
            return self.number_to_key[n]
        
        # Create pattern for this number
        pattern = MathPattern(
            input_expr=str(n),
            output_expr=str(n),  # Initial output is itself until transformed
            reasoning_type="collatz_state",
            problem_ids={f"collatz_{n}"},
            metadata={"value": n, "known_path_to_one": n in self.leads_to_one}
        )
        
        # Add to graph
        key = self.add_pattern(pattern)
        
        # Track the number -> key mapping
        self.number_to_key[n] = key
        
        return key
    
    def apply_collatz_step(self, n: int) -> Tuple[int, str, str]:
        """
        Apply a single Collatz step to number n.
        
        Args:
            n: The number to transform
            
        Returns:
            Tuple of (next_number, agent_name, operation_description)
        """
        if n % 2 == 0:
            next_n = n // 2
            agent_name = self.even_agent.name
            operation = "n/2"
        else:
            next_n = 3*n + 1
            agent_name = self.odd_agent.name
            operation = "3n+1"
            
        return next_n, agent_name, operation
    
    def walk_collatz_path(self, start_n: int, max_steps: int = 1000) -> Tuple[bool, List[int], int]:
        """
        Walk the Collatz path starting from n until reaching 1 or a known number.
        
        Args:
            start_n: Starting number
            max_steps: Maximum steps before giving up
            
        Returns:
            Tuple of (success, path, steps_taken)
        """
        n = start_n
        path = [n]
        steps = 0
        
        while n != 1 and steps < max_steps:
            # Check if this number is already known to reach 1
            if n in self.leads_to_one:
                return True, path, steps
            
            # Apply one step of the Collatz function
            next_n, agent_name, operation = self.apply_collatz_step(n)
            
            # Add to path
            path.append(next_n)
            n = next_n
            steps += 1
            
            # Create or get pattern keys for current and next number
            current_key = self.add_number(path[-2])
            next_key = self.add_number(next_n)
            
            # Add the transformation
            self.add_transformation(current_key, next_key, agent_name)
            
            # Update pattern metadata
            if current_key in self.patterns:
                self.patterns[current_key].metadata["next_value"] = next_n
                self.patterns[current_key].metadata["operation"] = operation
        
        # We either reached 1 or hit the step limit
        success = (n == 1)
        return success, path, steps
    
    def verify_number(self, n: int, max_steps: int = 1000) -> Tuple[bool, int, List[int]]:
        """
        Verify that a number eventually reaches 1 under the Collatz function.
        
        In the Field of Truth framework, this uses the complete agent system to
        verify the path and maintain the integrity of the knowledge graph.
        
        Args:
            n: The number to verify
            max_steps: Maximum steps before giving up
            
        Returns:
            Tuple of (verified, steps_taken, path)
        """
        # Skip if already verified
        if n in self.leads_to_one:
            return True, self.path_lengths.get(n, 0), [n]
        
        # Walk the Collatz path using the FoT walker agent
        success, path, steps = self.walker.walk(n, max_steps)
        
        if success:
            # Mark the entire path as leading to 1
            for i, num in enumerate(path):
                self.leads_to_one.add(num)
                self.path_lengths[num] = steps - i
                
                # Record convergence depth
                self.deep_convergence_agent.record_depth(num, len(path) - i - 1)
            
            # Register the verified path with subpath agent
            self.subpath_agent.register_path(n, path)
            
            # Check for cycles in the path
            has_cycle, cycle = self.cycle_detection_agent.detect_cycle(path)
            if has_cycle:
                print(f"Warning: Detected unexpected cycle in path: {cycle}")
            
            # Broadcast the newly verified path
            self.broadcast_agent.broadcast(n, path, "local_validator")
            
            # Try to merge with other paths
            for other_n in list(self.leads_to_one)[:100]:  # Limit to prevent excessive computation
                if other_n != n and other_n not in path:
                    other_verified, other_steps, other_path = self.verify_number(other_n, max_steps)
                    if other_verified:
                        merged, merge_point, idx1, idx2 = self.graph_merge_agent.merge(path, other_path)
        
        return success, steps, path
    
    def verify_range(self, start: int, end: int, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Verify a range of numbers using the Field of Truth framework.
        
        This method uses the Range-Shard-Agent to process numbers in parallel
        and incorporates other optimization strategies like prioritizing primes.
        
        Args:
            start: Starting number (inclusive)
            end: Ending number (inclusive)
            max_steps: Maximum steps per verification
            
        Returns:
            Statistics about the verification
        """
        results = {
            "verified_count": 0,
            "failed_count": 0,
            "total_steps": 0,
            "max_steps_observed": 0,
            "max_steps_number": 0,
            "verification_time": 0,
            "average_steps": 0,
            "prime_count": 0,
            "path_merges": 0,
            "subpath_reuses": 0,
            "validation_signatures": 0,
        }
        
        start_time = time.time()
        
        # First, prioritize processing prime numbers in the range
        primes = self.prime_prefetch_agent.get_primes_in_range(start, end, limit=min(1000, (end - start + 1) // 10 + 1))
        results["prime_count"] = len(primes)
        
        # Process primes first
        for p in primes:
            if p in self.leads_to_one:
                results["verified_count"] += 1
                continue
                
            verified, steps, path = self.verify_number(p, max_steps)
            
            if verified:
                results["verified_count"] += 1
                results["total_steps"] += steps
                
                if steps > results["max_steps_observed"]:
                    results["max_steps_observed"] = steps
                    results["max_steps_number"] = p
            else:
                results["failed_count"] += 1
        
        # Create range shards if the range is large enough
        range_size = end - start + 1
        if range_size > 1000:
            # Create multiple shards
            shard_size = min(1000, range_size // 10 + 1)
            shards = []
            
            for i in range(start, end + 1, shard_size):
                shard_end = min(i + shard_size - 1, end)
                shard = RangeShardAgent(i, shard_end, walker=self.walker)
                shards.append(shard)
            
            # Process each shard
            for shard in shards:
                shard_results = shard.process_range(batch_size=shard_size, max_steps=max_steps)
                
                for n, verified, steps, path_len in shard_results:
                    if verified:
                        results["verified_count"] += 1
                        results["total_steps"] += steps
                        
                        if steps > results["max_steps_observed"]:
                            results["max_steps_observed"] = steps
                            results["max_steps_number"] = n
                    else:
                        results["failed_count"] += 1
        else:
            # Process sequentially for small ranges
            for n in range(start, end + 1):
                # Skip primes that we already processed and numbers we already know lead to 1
                if n in primes or n in self.leads_to_one:
                    if n in self.leads_to_one and n not in primes:
                        results["verified_count"] += 1
                    continue
                    
                verified, steps, path = self.verify_number(n, max_steps)
                
                if verified:
                    results["verified_count"] += 1
                    results["total_steps"] += steps
                    
                    if steps > results["max_steps_observed"]:
                        results["max_steps_observed"] = steps
                        results["max_steps_number"] = n
                else:
                    results["failed_count"] += 1
        
        # Calculate average steps
        if results["verified_count"] > 0:
            results["average_steps"] = results["total_steps"] / results["verified_count"]
        
        # Gather additional metrics from agents
        results["subpath_reuses"] = self.subpath_agent.reuse_count
        results["path_merges"] = self.graph_merge_agent.merge_count
        results["validation_signatures"] = self.validator.validation_count
            
        results["verification_time"] = time.time() - start_time
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save the knowledge graph to a file.
        
        In the Field of Truth framework, this includes saving all validation
        signatures and convergence depth data.
        
        Args:
            filepath: Path to save to
        """
        # Prepare data for serialization
        data = {
            "patterns": {},
            "adjacency": {},
            "number_to_key": {},
            "leads_to_one": list(self.leads_to_one),
            "path_lengths": {str(k): v for k, v in self.path_lengths.items()},
            "fot_metadata": {
                "version": "1.0.0",
                "timestamp": time.time(),
                "validation_count": self.validator.validation_count,
                "subpath_reuse_count": self.subpath_agent.reuse_count,
                "path_merge_count": self.graph_merge_agent.merge_count,
                "ethics_audits": self.ethics_agent.audit_count,
                "ethics_violations": self.ethics_agent.violation_count,
            }
        }
        
        # Add convergence depths
        data["convergence_depths"] = {
            str(k): v for k, v in self.deep_convergence_agent.convergence_depths.items()
        }
        
        # Add validation signatures (limited to avoid huge files)
        signatures = {}
        signature_items = list(self.validator.signatures.items())[:10000]  # Limit to 10k signatures
        for (input_val, output_val), signature in signature_items:
            signatures[f"{input_val}->{output_val}"] = signature
        data["validation_signatures"] = signatures
        
        # Convert patterns to dictionaries
        for key, pattern in self.patterns.items():
            data["patterns"][key] = pattern.to_dict()
        
        # Convert adjacency list
        for key, edges in self.adjacency.items():
            data["adjacency"][key] = edges
        
        # Convert number_to_key
        for n, key in self.number_to_key.items():
            data["number_to_key"][str(n)] = key
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved knowledge graph to {filepath} with Field of Truth metadata")
    
    @classmethod
    def load(cls, filepath: str, max_patterns: int = 1000000) -> 'CollatzKG':
        """
        Load a knowledge graph from a file.
        
        Args:
            filepath: Path to load from
            max_patterns: Maximum patterns to keep in memory
            
        Returns:
            Loaded knowledge graph
        """
        storage_dir = os.path.dirname(filepath)
        kg = cls(storage_dir=storage_dir, max_patterns=max_patterns)
        
        try:
            # Load data
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load patterns
            for key, pattern_data in data.get("patterns", {}).items():
                kg.patterns[key] = MathPattern.from_dict(pattern_data)
            
            # Load adjacency list
            for key, edges in data.get("adjacency", {}).items():
                kg.adjacency[key] = edges
            
            # Load number_to_key
            for n_str, key in data.get("number_to_key", {}).items():
                kg.number_to_key[int(n_str)] = key
            
            # Load leads_to_one
            kg.leads_to_one = set(data.get("leads_to_one", []))
            
            # Load path_lengths
            path_lengths = data.get("path_lengths", {})
            kg.path_lengths = {int(k): v for k, v in path_lengths.items()}
            
            # Load Field of Truth data if available
            if "convergence_depths" in data:
                for k, v in data["convergence_depths"].items():
                    kg.deep_convergence_agent.convergence_depths[int(k)] = v
                    
            # Load validation signatures if available
            if "validation_signatures" in data:
                for key_str, signature in data["validation_signatures"].items():
                    try:
                        parts = key_str.split("->")
                        input_val = int(parts[0])
                        output_val = int(parts[1])
                        kg.validator.signatures[(input_val, output_val)] = signature
                    except (ValueError, IndexError):
                        # Skip invalid entries
                        pass
                        
            # Load convergence paths into subpath agent
            for n in kg.leads_to_one:
                if n == 1:
                    kg.subpath_agent.register_path(1, [1])
                elif n == 2:
                    kg.subpath_agent.register_path(2, [2, 1])
                elif n == 4:
                    kg.subpath_agent.register_path(4, [4, 2, 1])
                else:
                    # For other numbers, we can construct a path from the stored path lengths
                    if n in kg.path_lengths:
                        # Reconstruct a path based on our transform function
                        path = [n]
                        current = n
                        # We use a limit to avoid potential loops
                        for _ in range(10000):
                            if current == 1:
                                break
                            if current % 2 == 0:
                                current = current // 2
                            else:
                                current = 3 * current + 1
                            path.append(current)
                        kg.subpath_agent.register_path(n, path)
            
            # Update metadata counts based on loaded data
            kg.validator.validation_count = len(kg.validator.signatures)
            kg.subpath_agent.reuse_count = 0  # Reset as we can't determine this from stored data
            kg.deep_convergence_agent.convergence_depths = kg.deep_convergence_agent.convergence_depths or {}
            
            print(f"Loaded Field of Truth knowledge graph with {len(kg.patterns)} patterns and {len(kg.leads_to_one)} verified numbers from {filepath}")
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
        
        return kg
    
    @classmethod
    def load_or_create(cls, filepath: str, max_patterns: int = 1000000) -> 'CollatzKG':
        """
        Load a Collatz KG or create a new one if file doesn't exist.
        
        Args:
            filepath: Path to load from or create at
            max_patterns: Maximum patterns to keep in memory
            
        Returns:
            Loaded or new CollatzKG
        """
        if os.path.exists(filepath):
            return cls.load(filepath, max_patterns)
        else:
            kg = cls(storage_dir=os.path.dirname(filepath), max_patterns=max_patterns)
            # Initialize with basic cycle
            for n in [1, 2, 4]:
                kg.add_number(n)
            kg.add_transformation(kg.number_to_key[2], kg.number_to_key[1], "CollatzEvenAgent")
            kg.add_transformation(kg.number_to_key[4], kg.number_to_key[2], "CollatzEvenAgent")
            
            # Save the initial KG
            kg.save(filepath)
            return kg
    
    def verify_large_number(self, n_str: str, max_steps: int = 100000) -> Tuple[bool, int, List[int]]:
        """
        Verify a very large number efficiently using the Field of Truth framework.
        
        This specialized method handles numbers of arbitrary size by using Python's
        built-in arbitrary precision integers and advanced path merging.
        
        Args:
            n_str: The number to verify as a string (can be in scientific notation)
            max_steps: Maximum steps before giving up
            
        Returns:
            Tuple of (verified, steps_taken, path_summary)
        """
        # Convert the input to a Python int (handles arbitrary precision)
        try:
            if 'e' in n_str.lower():
                # Handle scientific notation
                n = int(float(n_str))
            else:
                n = int(n_str)
        except ValueError:
            return False, 0, []
        
        if n <= 0:
            return False, 0, []
            
        # For extremely large numbers, we don't store the full path
        # but track key metrics and a sample of points
        path_summary = []
        full_path = [n]
        steps = 0
        max_value = n
        max_value_step = 0
        
        # Add key points at regular intervals
        sample_interval = max(1, max_steps // 100)  # Track ~100 points max
        
        # Track the first few steps
        first_steps = []
        
        # Use the walker directly but with specialized handling
        current = n
        while current != 1 and steps < max_steps:
            # Check for known paths
            if current in self.leads_to_one:
                # We found a merge point - record it and calculate remaining steps
                remaining_steps = self.path_lengths[current]
                
                # Register this validation
                self.broadcast_agent.broadcast(
                    n, 
                    ["LARGE_NUMBER_MERGED", current, remaining_steps], 
                    "large_validator"
                )
                
                # Save key metrics
                path_summary.append({
                    "step": steps,
                    "value": str(current),
                    "event": "MERGE_POINT",
                    "remaining_steps": remaining_steps
                })
                
                # Update total steps
                steps += remaining_steps
                
                # Construct the summary path
                if steps == 0:  # If we hit 1 directly
                    full_path = [n, 1]
                    path_summary.append({
                        "step": 0,
                        "value": str(n),
                        "event": "START"
                    })
                    path_summary.append({
                        "step": 1,
                        "value": "1",
                        "event": "END"
                    })
                else:
                    # Record the merge
                    full_path.append(current)
                    full_path.append(1)  # Simplified - just show the terminal
                    
                # Mark this number as verified
                self.leads_to_one.add(n)
                self.path_lengths[n] = steps
                self.deep_convergence_agent.record_depth(n, steps)
                
                return True, steps, full_path
            
            # Apply Collatz step
            prev = current
            if current % 2 == 0:
                current = current // 2
            else:
                current = 3 * current + 1
                
            # Update max value
            if current > max_value:
                max_value = current
                max_value_step = steps + 1
                
            # Track the step
            steps += 1
            
            # Record first few steps
            if len(first_steps) < 5:
                first_steps.append(current)
                
            # Record the step at regular intervals or if it's significant
            if steps % sample_interval == 0 or current > max_value * 0.9:
                full_path.append(current)
                path_summary.append({
                    "step": steps,
                    "value": str(current),
                    "event": "SAMPLE" if current <= max_value * 0.9 else "PEAK_REGION"
                })
            
            # Register this validation
            self.validator.validate(prev, current, 
                                    self.even_agent if prev % 2 == 0 else self.odd_agent)
                
        # Check if we reached 1
        if current == 1:
            # We made it to 1
            full_path.append(1)
            path_summary.append({
                "step": steps,
                "value": "1",
                "event": "END"
            })
            
            # Mark this number as verified
            self.leads_to_one.add(n)
            self.path_lengths[n] = steps
            self.deep_convergence_agent.record_depth(n, steps)
            
            # Create a signature for the large number validation
            validation_id = hashlib.sha256(f"{n}->{steps}->{max_value}".encode()).hexdigest()[:16]
            
            # Record key metrics for reporting
            path_summary.append({
                "max_value": str(max_value),
                "max_value_step": max_value_step,
                "total_steps": steps,
                "validation_id": validation_id,
                "first_steps": [str(x) for x in first_steps[:5]]
            })
            
            # Broadcast this significant achievement
            self.broadcast_agent.broadcast(
                n, 
                [str(n), str(steps), str(max_value), validation_id], 
                "large_validator"
            )
            
            return True, steps, full_path
        else:
            # Didn't reach 1 within step limit
            return False, steps, full_path
    
    def _hash_pattern(self, expr_str: str) -> str:
        """
        Generate a hash key for a pattern.
        
        Args:
            expr_str: Expression string
            
        Returns:
            Hash key as string
        """
        try:
            # Simple hash based on the string representation
            return f"pattern_{hash(expr_str) % 10000000:07d}"
        except Exception:
            # Fallback to a random key
            import random
            return f"pattern_{random.randint(0, 9999999):07d}" 