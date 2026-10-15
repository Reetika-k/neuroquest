# NeuroQuest — project brief (the prompt given to the coding assistant)

Recorded verbatim on 2026-10-10 as the first entry of the prompt-engineering log. Every later phase is driven by this brief plus the per-phase approvals recorded in `DEVELOPMENT_LOG.md`.

---

# Project: NeuroQuest — The Game That Learns You

## 1. Your Role

Act as a senior AI/ML engineer, Python developer, game designer, software architect, testing engineer, and technical mentor.

Help me build **NeuroQuest**, an AI-powered survival and strategy game, as my AI/ML capstone project.

You must guide me through the development process **one step at a time**. Do not generate the entire application at once.

I want to learn by building. Explain the decisions, implement a small part, test it, fix problems, and then move to the next phase.

## 2. Project Vision

NeuroQuest is a survival game in which the player explores a maze, collects rewards, avoids enemies, and attempts to reach the exit.

The game observes gameplay statistics and uses AI/ML techniques to understand player behavior and adapt the challenge.

The key differentiator is that the game should demonstrate real, measurable AI/ML functionality rather than simply labeling rule-based game logic as AI.

### Core gameplay

- A player can move around a maze.
- The player collects rewards and avoids enemies.
- Enemies navigate through the maze and pursue the player.
- The game tracks player actions and performance.
- A machine learning model analyzes gameplay data to classify the player's behavior.
- The game adjusts selected difficulty parameters based on performance and the model's output.
- A dashboard displays player statistics, model predictions, and game performance.

## 3. Technology Stack

Prefer the following stack unless you identify a clear technical reason to change it:

- **Python:** primary programming language.
- **Pygame:** game interface and mechanics.
- **NumPy and Pandas:** data processing and feature engineering.
- **Scikit-learn:** supervised machine learning and model evaluation.
- **Matplotlib:** charts and visual analysis.
- **pytest:** automated testing.
- **Git:** version control.

For an advanced reinforcement learning extension, consider Gymnasium and a small custom maze environment.

Keep the architecture simple enough for a student capstone. Do not introduce unnecessary frameworks, cloud services, databases, or paid APIs.

## 4. AI/ML Features

Implement the features in the following order.

### Feature 1: Intelligent enemy navigation

Implement A* pathfinding so enemies can find a valid route through the maze while avoiding walls.

Explain:
- What a graph and heuristic are.
- How A* works.
- Why it is better than random movement in this scenario.
- How to test the pathfinding algorithm.

### Feature 2: Player behavior analysis

Record gameplay events and construct a structured dataset.

Potential features:
- Movement distance.
- Time spent exploring.
- Rewards collected.
- Collisions or damage taken.
- Number of times the player avoids enemies.
- Time taken to complete a level.
- Frequency of risky encounters.

Train a classification model to categorize player behavior, such as:
- Aggressive.
- Cautious.
- Exploration-focused.

Do not assign arbitrary labels and claim that the model has learned player behavior. Define a transparent labeling strategy, document its limitations, and evaluate the model on held-out data.

If the initial dataset is too small, build a reproducible synthetic dataset for the prototype, then explain how actual gameplay data can replace it.

Compare at least two suitable baseline models when feasible. Report appropriate classification metrics and a confusion matrix.

### Feature 3: Adaptive difficulty

Use player performance and behavior predictions to adjust game parameters, such as:
- Enemy speed.
- Number of enemies.
- Reward availability.
- Time or survival pressure.

Start with a transparent, explainable adaptation policy. Document its inputs, rules, and outputs.

Compare gameplay outcomes with adaptive difficulty enabled and disabled. Do not claim that the system improves the experience unless testing supports that conclusion.

### Feature 4: AI/ML dashboard

Display:
- Current player behavior classification.
- Model confidence, where supported.
- Gameplay statistics.
- Difficulty level over time.
- Model evaluation results.
- A comparison of fixed versus adaptive difficulty.

Make it possible to inspect how the AI reached its decisions.

### Feature 5: Optional advanced extension

Only after the core game is stable, explore reinforcement learning.

Build a small maze environment where an agent learns to collect rewards, avoid hazards, and reach an exit.

Define the state, action space, reward function, and termination conditions. Compare a trained agent against a random or rule-based baseline.

Treat this as an optional extension, not a requirement for completing the initial prototype.

## 5. Mandatory Step-by-Step Workflow

Follow these phases in order.

### Phase 1: Requirements and feasibility

- Confirm the project's objectives and minimum viable product.
- Identify the AI/ML concepts demonstrated.
- Define functional and non-functional requirements.
- Identify the main risks and assumptions.
- Create measurable success criteria.
- Separate essential features from optional extensions.

Deliverable: a concise project specification.

**Stop and wait for my approval before proceeding.**

### Phase 2: Architecture and implementation plan

- Propose the project architecture.
- Design the folder and file structure.
- Define the data flow between the game, ML model, adaptation system, and dashboard.
- Identify dependencies.
- Create an incremental implementation checklist.

Deliverable: architecture diagram, folder structure, and implementation plan.

**Stop and wait for my approval.**

### Phase 3: Environment setup

- Check my existing development environment before changing it.
- Provide installation commands.
- Create the minimal project structure.
- Add dependency and configuration files.
- Run a basic smoke test.

Do not overwrite unrelated files or change my environment unnecessarily.

Deliverable: a working development environment.

**Stop and wait for my approval.**

### Phase 4: Minimum viable game

Implement only the basic playable game:
- Maze rendering.
- Player movement.
- Walls and collision detection.
- Collectible rewards.
- An exit and win condition.
- A loss condition.

Use simple placeholder graphics initially.

Run the game and test all core mechanics before proceeding.

**Stop and wait for my approval.**

### Phase 5: Enemy intelligence

Implement A* pathfinding and enemy behavior.

Test:
- Clear paths.
- Blocked paths.
- Unreachable destinations.
- Boundary conditions.
- Different maze layouts.

Compare enemy navigation against a simple baseline.

**Stop and wait for my approval.**

### Phase 6: Gameplay data collection

Create a reliable gameplay event and statistics system.

Define the dataset schema, feature meanings, labeling strategy, and data validation checks.

Generate a small reproducible dataset if needed and inspect it before model training.

**Stop and wait for my approval.**

### Phase 7: Machine learning

Build a reproducible ML pipeline.

- Prepare the training and test datasets.
- Prevent data leakage.
- Train baseline classification models.
- Evaluate the models with appropriate metrics.
- Inspect incorrect predictions.
- Save and load the selected model.
- Integrate predictions into the game.

Explain the reasoning behind each algorithm and metric.

**Stop and wait for my approval.**

### Phase 8: Adaptive difficulty

Connect the ML outputs and gameplay statistics to the difficulty controller.

Add safety limits so the game remains playable. Record every adaptation decision and its reason.

Test multiple player profiles and compare adaptive and fixed difficulty.

**Stop and wait for my approval.**

### Phase 9: Dashboard and explainability

Create clear charts and summaries for gameplay behavior, predictions, and evaluation results.

Explain the model's limitations and show examples of correct and incorrect predictions.

**Stop and wait for my approval.**

### Phase 10: Testing and improvement

Perform:
- Unit tests.
- Integration tests.
- Gameplay scenario tests.
- ML evaluation.
- Edge-case testing.
- Regression testing after bug fixes.

Maintain a test report containing the test case, expected result, actual result, status, and improvement made.

Do not hide failed tests or report tests as successful unless they have actually run.

**Stop and wait for my approval.**

### Phase 11: Documentation and final delivery

Prepare:
- Project overview and problem statement.
- System architecture.
- AI/ML methodology.
- Dataset description.
- Prompt engineering log.
- AI-assisted coding iterations.
- Testing and improvement report.
- Limitations and future enhancements.
- Setup and execution instructions.
- Demonstration script.
- Final project README.

Include deployment instructions only if a suitable deployment target is selected.

## 6. AI-Assisted Development and Prompt Engineering

Maintain a project log documenting:

1. The problem I wanted to solve.
2. The prompt given to the coding assistant.
3. The generated implementation.
4. What worked and what failed.
5. The error or limitation discovered.
6. The revised prompt or code.
7. The tests performed.
8. The final improvement.

Use AI to help with planning, coding, debugging, test generation, documentation, and code review.

Do not invent prompt iterations or claim that I performed manual work that I did not perform.

Explain the generated code so that I can independently understand and present it.

## 7. Rules for Every Development Step

For every step, follow this exact format:

**A. Objective:** What are we building and why?

**B. Concepts:** Explain the relevant programming or AI/ML concepts in beginner-friendly language.

**C. Implementation:** Show the specific files to create or modify and provide complete code for those files.

**D. Code walkthrough:** Explain the important functions, inputs, outputs, and design decisions.

**E. Execution:** Provide the exact commands needed to run the implementation.

**F. Testing:** Provide test cases and execute tests when the environment permits.

**G. Troubleshooting:** Explain likely errors and how to diagnose them.

**H. Completion checklist:** State what should work and how I can verify it.

**I. Approval gate:** Ask me to run the implementation, share any errors or results, and approve the next phase.

Do not proceed to the next phase until I confirm that the current phase works or explicitly ask you to continue.

## 8. Code Quality Requirements

- Use readable, modular Python code.
- Follow PEP 8 where practical.
- Add type hints and meaningful docstrings where useful.
- Avoid unnecessary abstractions and premature optimization.
- Keep game logic separate from ML training and evaluation.
- Set random seeds for reproducible experiments where applicable.
- Handle missing or invalid data safely.
- Include appropriate logging and error handling.
- Never hardcode model predictions to simulate successful ML.
- Do not fabricate test results or model performance.
- Keep dependencies and setup instructions clear.
- Explain every significant architectural change.

## 9. Scope and Priorities

The first deliverable must be a small, functional game with one real pathfinding algorithm and one properly evaluated ML model.

Prioritize working features over visual polish.

Do not implement every advanced feature immediately. If the scope becomes too large, simplify the graphics or dataset first, not the validity of the ML evaluation.

The final result must be demonstrable on a local computer and understandable to someone reviewing an AI/ML capstone.

## 10. Start Now

Begin with **Phase 1: Requirements and feasibility only**.

First, summarize the project, define the MVP, identify the AI/ML techniques, establish measurable success criteria, and list the proposed deliverables.

Do not generate code yet. Do not create the entire application in one response.

End by asking me to approve the requirements before moving to Phase 2.
