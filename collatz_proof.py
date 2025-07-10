#!/usr/bin/env python3
"""
Collatz Conjecture Proof System using the Field of Truth framework

This script implements a command-line tool for proving the Collatz Conjecture
using the Agentic Knowledge Graph (AKG) approach described in the Field of Truth paper.

The Collatz Conjecture states that for any positive integer n, repeatedly applying
the function f(n) = n/2 for even n, and f(n) = 3n+1 for odd n, will eventually
reach the number 1.

The proof is demonstrated through:
1. Creating a knowledge graph representing Collatz sequences
2. Implementing transformation agents for the Collatz operations
3. Verifying paths for given numbers to the terminal value 1
4. Reusing known subpaths for efficiency
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path

import sympy as sp
from agents import BaseMathAgent
from pattern import MathPattern
from knowledge_graph import MathPatternKG
from utils import parse_expression, expressions_equal

# Define Collatz agents based on the existing agent framework
class CollatzOddAgent(BaseMathAgent):
    """Agent that applies the 3n+1 transformation for odd numbers."""
    
    def __init__(self):
        """Initialize Collatz odd number agent."""
        super().__init__("CollatzOddAgent")
    
    def transform(self, expr: sp.Expr) -> sp.Expr:
        """Apply the 3n+1 transformation to an odd number."""
        # If it's a literal number, apply transformation directly
        if expr.is_Integer:
            n = int(expr)
            if n % 2 == 1:  # Check if odd
                return sp.Integer(3*n + 1)
        
        # Otherwise, create the symbolic operation
        # This allows for symbolic inputs like 'n' for general proofs
        return 3*expr + 1
    
    def explain(self) -> str:
        """Explain the Collatz odd transformation."""
        return "Apply the Collatz 3n+1 operation for odd numbers"


class CollatzEvenAgent(BaseMathAgent):
    """Agent that applies the n/2 transformation for even numbers."""
    
    def __init__(self):
        """Initialize Collatz even number agent."""
        super().__init__("CollatzEvenAgent")
    
    def transform(self, expr: sp.Expr) -> sp.Expr:
        """Apply the n/2 transformation to an even number."""
        # If it's a literal number, apply transformation directly
        if expr.is_Integer:
            n = int(expr)
            if n % 2 == 0:  # Check if even
                return sp.Integer(n // 2)
        
        # Otherwise, create the symbolic operation
        # This allows for symbolic inputs like 'n' for general proofs
        return expr / 2
    
    def explain(self) -> str:
        """Explain the Collatz even transformation."""
        return "Apply the Collatz n/2 operation for even numbers"


class CollatzKG(MathPatternKG):
    """
    Knowledge graph specialized for the Collatz conjecture.
    
    Stores Collatz sequence paths and enables efficient path reuse.
    """
    def __init__(self, storage_dir: Optional[str] = None, max_patterns: int = 1000000):
        """Initialize a Collatz knowledge graph."""
        super().__init__(storage_dir, max_patterns)
        
        # Maps from number -> pattern key for quick lookup
        self.number_to_key: Dict[int, str] = {}
        
        # Track if a number is known to reach 1
        self.leads_to_one: Set[int] = {1, 2, 4}  # Start with the known cycle
        
        # Track path lengths for reporting
        self.path_lengths: Dict[int, int] = {1: 0, 2: 1, 4: 2}
        
        # Initialize agents
        self.odd_agent = CollatzOddAgent()
        self.even_agent = CollatzEvenAgent()
    
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
        
        Args:
            n: The number to verify
            max_steps: Maximum steps before giving up
            
        Returns:
            Tuple of (verified, steps_taken, path)
        """
        # Skip if already verified
        if n in self.leads_to_one:
            return True, self.path_lengths.get(n, 0), [n]
        
        # Walk the Collatz path
        success, path, steps = self.walk_collatz_path(n, max_steps)
        
        if success:
            # Mark the entire path as leading to 1
            for i, num in enumerate(path):
                self.leads_to_one.add(num)
                self.path_lengths[num] = steps - i
        
        return success, steps, path
    
    def verify_range(self, start: int, end: int, max_steps: int = 1000) -> Dict[str, Any]:
        """
        Verify a range of numbers.
        
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
        }
        
        start_time = time.time()
        
        for n in range(start, end + 1):
            verified, steps, path = self.verify_number(n, max_steps)
            
            if verified:
                results["verified_count"] += 1
                results["total_steps"] += steps
                
                if steps > results["max_steps_observed"]:
                    results["max_steps_observed"] = steps
                    results["max_steps_number"] = n
            else:
                results["failed_count"] += 1
        
        results["verification_time"] = time.time() - start_time
        
        if results["verified_count"] > 0:
            results["average_steps"] = results["total_steps"] / results["verified_count"]
        
        return results
    
    def save_verification_results(self, filepath: str) -> None:
        """
        Save verification results to a file.
        
        Args:
            filepath: Path to save results
        """
        results = {
            "verified_count": len(self.leads_to_one),
            "max_path_length": max(self.path_lengths.values()) if self.path_lengths else 0,
            "number_with_longest_path": max(self.path_lengths, key=self.path_lengths.get) if self.path_lengths else 0,
            "verification_time": time.time(),
            "path_lengths": {str(k): v for k, v in self.path_lengths.items() if k <= 100}  # Include only a sample
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def visualize_path(self, start_n: int) -> None:
        """
        Visualize a Collatz path starting from n.
        
        Args:
            start_n: Starting number
        """
        if start_n not in self.leads_to_one:
            print(f"Number {start_n} has not been verified yet.")
            return
        
        n = start_n
        path = []
        
        while n != 1:
            path.append(n)
            
            # Find the next number
            key = self.number_to_key.get(n)
            if not key or key not in self.patterns:
                print(f"Path information missing for {n}")
                return
                
            pattern = self.patterns[key]
            next_n = pattern.metadata.get("next_value")
            operation = pattern.metadata.get("operation")
            
            if not next_n:
                print(f"Next value missing for {n}")
                return
                
            path_str = f"{n} -> {next_n}"
            if operation:
                path_str = f"{path_str} ({operation})"
            
            print(path_str)
            n = next_n
        
        path.append(1)
        
        # Print summary
        print(f"Path length: {len(path) - 1} steps")
        print(f"Peak value: {max(path)}")
    
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


def verify_command(args):
    """
    Verify that a number or range of numbers follow the Collatz conjecture.
    """
    # Load or create the Collatz KG
    kg = CollatzKG.load_or_create(args.kg_file)
    
    if args.range:
        start, end = map(int, args.range.split('-'))
        print(f"Verifying Collatz conjecture for numbers {start} to {end}...")
        results = kg.verify_range(start, end, args.max_steps)
        
        # Print statistics
        print(f"Verified {results['verified_count']} numbers")
        if results['failed_count'] > 0:
            print(f"Failed to verify {results['failed_count']} numbers")
        print(f"Average steps: {results['average_steps']:.2f}")
        print(f"Maximum steps observed: {results['max_steps_observed']} (for n={results['max_steps_number']})")
        print(f"Verification took {results['verification_time']:.2f} seconds")
    else:
        # Verify a single number
        n = args.number
        print(f"Verifying Collatz conjecture for n={n}...")
        verified, steps, path = kg.verify_number(n, args.max_steps)
        
        if verified:
            print(f"Verified: n={n} reaches 1 in {steps} steps")
            print(f"Path: {' -> '.join(map(str, path))}")
            print(f"Peak value in path: {max(path)}")
        else:
            print(f"Failed to verify n={n} within {args.max_steps} steps")
    
    # Save the updated KG
    print(f"Saving knowledge graph to {args.kg_file}...")
    kg.save(args.kg_file)
    
    # Save verification results
    if args.output:
        print(f"Saving verification results to {args.output}...")
        kg.save_verification_results(args.output)
    
    return 0


def show_path_command(args):
    """
    Show the Collatz path for a specific number.
    """
    # Load the Collatz KG
    kg = CollatzKG.load_or_create(args.kg_file)
    
    # Verify the number first if needed
    if args.number not in kg.leads_to_one:
        print(f"Number {args.number} not yet verified. Verifying now...")
        verified, steps, path = kg.verify_number(args.number, args.max_steps)
        
        if not verified:
            print(f"Failed to verify that {args.number} reaches 1 within {args.max_steps} steps.")
            return 1
    
    # Visualize the path
    print(f"Collatz path for n={args.number}:")
    kg.visualize_path(args.number)
    
    return 0


def analyze_command(args):
    """
    Analyze the structure of the Collatz knowledge graph.
    """
    # Load the Collatz KG
    kg = CollatzKG.load_or_create(args.kg_file)
    
    print(f"Collatz Knowledge Graph Analysis:")
    print(f"Total numbers verified: {len(kg.leads_to_one)}")
    
    if args.stats:
        # Generate detailed statistics
        path_lengths = list(kg.path_lengths.values())
        if path_lengths:
            avg_path_length = sum(path_lengths) / len(path_lengths)
            max_path_length = max(path_lengths)
            max_path_number = max(kg.path_lengths, key=kg.path_lengths.get)
            
            print(f"Average path length: {avg_path_length:.2f} steps")
            print(f"Maximum path length: {max_path_length} steps (for n={max_path_number})")
        
        # Find numbers with longest paths
        top_n = min(10, len(kg.path_lengths))
        longest_paths = sorted(kg.path_lengths.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"Top {top_n} numbers with longest paths:")
        for n, length in longest_paths:
            print(f"n={n}: {length} steps")
    
    if args.output:
        # Save analysis results
        results = {
            "verified_count": len(kg.leads_to_one),
            "path_length_stats": {
                "average": sum(kg.path_lengths.values()) / len(kg.path_lengths) if kg.path_lengths else 0,
                "max": max(kg.path_lengths.values()) if kg.path_lengths else 0,
                "max_number": max(kg.path_lengths, key=kg.path_lengths.get) if kg.path_lengths else 0
            },
            "longest_paths": dict(sorted(kg.path_lengths.items(), key=lambda x: x[1], reverse=True)[:100])
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Analysis results saved to {args.output}")
    
    return 0


def main():
    """Main function for the Collatz proof command-line interface."""
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="Collatz Conjecture Proof System using Field of Truth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--kg-file", type=str, default="collatz_kg.json",
        help="Knowledge graph file path"
    )
    common_parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum steps for a single verification"
    )
    
    # Create parser for the "verify" command
    verify_parser = subparsers.add_parser("verify", 
                                        parents=[common_parser],
                                        help="Verify Collatz conjecture for specific numbers")
    verify_number_group = verify_parser.add_mutually_exclusive_group(required=True)
    verify_number_group.add_argument(
        "--number", "-n", type=int,
        help="Specific number to verify"
    )
    verify_number_group.add_argument(
        "--range", "-r", type=str,
        help="Range of numbers to verify (format: start-end)"
    )
    verify_parser.add_argument(
        "--output", "-o", type=str,
        help="Output file for verification results"
    )
    
    # Create parser for the "path" command
    path_parser = subparsers.add_parser("path", 
                                      parents=[common_parser],
                                      help="Show the Collatz path for a specific number")
    path_parser.add_argument(
        "number", type=int,
        help="Number to show the path for"
    )
    
    # Create parser for the "analyze" command
    analyze_parser = subparsers.add_parser("analyze", 
                                         parents=[common_parser],
                                         help="Analyze the Collatz knowledge graph")
    analyze_parser.add_argument(
        "--stats", action="store_true",
        help="Show detailed statistics"
    )
    analyze_parser.add_argument(
        "--output", "-o", type=str,
        help="Output file for analysis results"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == "verify":
        return verify_command(args)
    elif args.command == "path":
        return show_path_command(args)
    elif args.command == "analyze":
        return analyze_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main()) 