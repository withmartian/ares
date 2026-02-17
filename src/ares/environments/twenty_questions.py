"""Twenty Questions environment for ARES.

This environment implements the 20 questions game where an agent asks yes/no questions
to guess a hidden object. An LLM-based oracle answers the questions.
"""

import logging
import random
import time
from types import TracebackType
from typing import Self

import frozendict

from ares.environments import base
from ares.experiment_tracking import stat_tracker
from ares.llms import chat_completions_compatible
from ares.llms import request
from ares.llms import response

_LOGGER = logging.getLogger(__name__)

# Oracle system prompt template - instructs the LLM to answer yes/no questions
# Note: ArCHeR uses a single-line prompt string for this template
ORACLE_PROMPT_TEMPLATE = (
    'You are playing a game called twenty questions with me. The rule of twenty question is that '
    'you are given a hidden word, and I am guessing what the word is within twenty questions. '
    'For every question, if it is an invalid question, you should answer "Invalid Question.". '
    'For any valid question, you should answer either "Yes." or "No.". '
    'Now the hidden word given to you is "{word}", and the question for the current round is '
    '"{question}". Your response is:'
)

# Default object dictionary from ArCHeR paper - 16 categories, ~160 objects
DEFAULT_OBJECT_DICT = frozendict.frozendict(
    {
        "Sports": (
            "Basketball",
            "Football",
            "Baseball",
            "Soccer ball",
            "Golf ball",
            "Tennis ball",
            "Volleyball",
            "Tennis racket",
            "Baseball bat",
            "Helmet",
        ),
        "Animals": ("Cat", "Dog", "Horse", "Cow", "Sheep", "Rabbit", "Lion", "Tiger", "Bear", "Elephant"),
        "Fruits": (
            "Apple",
            "Banana",
            "Orange",
            "Strawberry",
            "Grape",
            "Watermelon",
            "Pineapple",
            "Mango",
            "Cantaloupe",
            "Peach",
        ),
        "Vehicles": (
            "Car",
            "Truck",
            "Motorcycle",
            "Boat",
            "Airplane",
            "Train",
            "Bus",
            "Helicopter",
            "Scooter",
            "Ship",
        ),
        "Clothes": ("Shirt", "Pants", "Jacket", "Dress", "Skirt", "Belt", "Shoes", "Boots", "Socks", "Hat", "Scarf"),
        "Electronics": (
            "Computer",
            "Smartphone",
            "Television",
            "Headphones",
            "Monitor",
            "Camera",
            "Microwave",
            "Refrigerator",
            "Blender",
            "Keyboard",
        ),
        "Musical Instruments": (
            "Piano",
            "Guitar",
            "Drum",
            "Violin",
            "Saxophone",
            "Flute",
            "Trumpet",
            "Clarinet",
            "Harp",
            "Trombone",
        ),
        "Furniture": (
            "Chair",
            "Table",
            "Bed",
            "Desk",
            "Couch",
            "Dresser",
            "Bookcase",
            "Nightstand",
            "Mattress",
            "Pillow",
        ),
        "Office Supplies": (
            "Pen",
            "Paper",
            "Stapler",
            "Printer",
            "Calculator",
            "Battery",
            "Toothbrush",
            "Toothpaste",
            "Pencil",
            "Sharpie",
            "Scissors",
            "Key",
            "Diary",
            "Calendar",
        ),
        "Vegetables": (
            "Carrot",
            "Potato",
            "Broccoli",
            "Tomato",
            "Onion",
            "Spinach",
            "Corn",
            "Peas",
            "Celery",
            "Cucumber",
        ),
        "Art": ("Painting", "Paintbrush", "Canvas", "Eraser", "Marker", "Glue", "Sculpture"),
        "Kitchen Tools": ("Knife", "Spoon", "Fork", "Plate", "Bowl", "Pot", "Pan", "Cup", "Chopsticks", "Whisk"),
        "Nature": (
            "Rock",
            "Tree",
            "Bush",
            "Mountain",
            "Forest",
            "Ocean",
            "Sea",
            "Lake",
            "River",
            "Meteorite",
            "Cactus",
        ),
        "Toys": ("Lego", "Doll", "Kite", "Puzzle"),
        "Jewelry": ("Earrings", "Necklace", "Bracelet", "Ring", "Brooch", "Hairclip", "Pendant", "Watch", "Locket"),
        "Garden Supplies": ("Gloves", "Shovel", "Rake", "Watering can", "Lawn mower"),
        "Tools": ("Hammer", "Screwdriver", "Wrench", "Saw", "Pliers", "Drill"),
    }
)

# Flatten the object dictionary into a single tuple
DEFAULT_OBJECT_LIST = tuple(obj for objects in DEFAULT_OBJECT_DICT.values() for obj in objects)

# Simplified 10-object subset used in ArCHeR LLM experiments
SIMPLE_OBJECT_LIST = ("Football", "Dog", "Banana", "Truck", "Pants", "Computer", "Piano", "Chair", "Pen", "Scissors")


class TwentyQuestionsEnvironment(base.Environment[response.LLMResponse, request.LLMRequest, float, float]):
    """Environment for twenty questions game using an LLM-based oracle."""

    def __init__(
        self,
        *,
        objects: tuple[str, ...] | list[str] | None = None,
        oracle_model: str = "gpt-4o-mini",  # Cheap model for oracle
        step_limit: int = 20,
        prefix: str = "twenty_questions",
        tracker: stat_tracker.StatTracker | None = None,
    ):
        """Initialize the Twenty Questions environment.

        Args:
            objects: Tuple or list of possible objects to guess. If None, uses DEFAULT_OBJECT_LIST.
            oracle_model: Model to use for the oracle (default: gpt-4o-mini, very cheap).
            step_limit: Maximum number of questions allowed (default: 20).
            prefix: Prefix for stat tracking.
            tracker: Optional stat tracker for metrics.
        """
        # Convert to tuple if list is provided for consistency with immutable defaults
        self._objects = tuple(objects) if objects is not None else DEFAULT_OBJECT_LIST
        self._oracle_model = oracle_model
        self._step_limit = step_limit
        self._prefix = prefix
        self._tracker = tracker if tracker is not None else stat_tracker.NullStatTracker()

        # Create oracle LLM client
        self._oracle_client = chat_completions_compatible.ChatCompletionCompatibleLLMClient(model=oracle_model)

        # Episode state
        self._is_active = False
        self._hidden_object: str | None = None
        self._conversation_history: list[str] = []
        self._step_count = 0
        self._requires_reset = False

    async def reset(self) -> base.TimeStep[request.LLMRequest, float, float]:
        """Start a new episode by selecting a random object."""
        reset_start_time = time.time()
        self._assert_active()

        _LOGGER.debug("[%d] Resetting twenty questions environment.", id(self))

        self._step_count = 0
        self._requires_reset = False
        self._conversation_history = []

        # Select a random object
        self._hidden_object = random.choice(self._objects)
        _LOGGER.info("[%d] Selected hidden object: %s", id(self), self._hidden_object)

        # Create initial observation - just the game instructions
        initial_prompt = (
            "Let's play Twenty Questions! I'm thinking of an object. "
            f"You have {self._step_limit} questions to guess what it is. "
            "Ask yes/no questions to narrow down the possibilities. "
            "When you think you know the answer, ask 'Is it [object]?'"
        )

        observation = request.LLMRequest(
            messages=[
                request.UserMessage(role="user", content=initial_prompt),
            ]
        )

        reset_end_time = time.time()
        self._tracker.scalar(f"{self._prefix}/reset", reset_end_time - reset_start_time)

        return base.TimeStep(step_type="FIRST", reward=None, discount=None, observation=observation)

    async def step(self, action: response.LLMResponse) -> base.TimeStep[request.LLMRequest, float, float]:
        """Process agent's question and get oracle's answer."""
        step_start_time = time.time()
        self._assert_active()

        _LOGGER.debug("[%d] Stepping twenty questions environment.", id(self))

        if self._requires_reset:
            raise RuntimeError("Environment must be reset.")

        self._step_count += 1

        # Extract the question from the agent's response
        question = self._extract_question_from_response(action)
        _LOGGER.debug("[%d] Agent question: %s", id(self), question)

        # Check if the agent is making a guess
        is_correct_guess = self._check_if_correct_guess(question)

        if is_correct_guess:
            # Agent guessed correctly!
            _LOGGER.info("[%d] Agent guessed correctly: %s", id(self), self._hidden_object)
            self._conversation_history.append(f"Q{self._step_count}: {question}")
            self._conversation_history.append("A: Yes! You guessed correctly!")

            # Terminal state with positive reward
            self._requires_reset = True
            reward = 0.0  # Success reward (0 is better than negative)
            observation_content = (
                "\n".join(self._conversation_history)
                + f"\n\nYou win! The object was {self._hidden_object}."
            )

            observation = request.LLMRequest(
                messages=[
                    request.UserMessage(role="user", content=observation_content),
                ]
            )

            step_end_time = time.time()
            self._tracker.scalar(f"{self._prefix}/step", step_end_time - step_start_time)
            self._tracker.scalar(f"{self._prefix}/success", 1.0)

            return base.TimeStep(step_type="LAST", reward=reward, discount=0.0, observation=observation)

        # Get oracle's answer
        oracle_answer = await self._get_oracle_answer(question)
        _LOGGER.debug("[%d] Oracle answer: %s", id(self), oracle_answer)

        # Update conversation history
        self._conversation_history.append(f"Q{self._step_count}: {question}")
        self._conversation_history.append(f"A: {oracle_answer}")

        # Step penalty
        reward = -1.0

        # Check if we've hit the step limit
        if self._step_count >= self._step_limit:
            _LOGGER.info("[%d] Step limit reached. Agent failed to guess: %s", id(self), self._hidden_object)
            self._requires_reset = True
            observation_content = (
                "\n".join(self._conversation_history)
                + f"\n\nYou've run out of questions! The object was {self._hidden_object}."
            )

            observation = request.LLMRequest(
                messages=[
                    request.UserMessage(role="user", content=observation_content),
                ]
            )

            step_end_time = time.time()
            self._tracker.scalar(f"{self._prefix}/step", step_end_time - step_start_time)
            self._tracker.scalar(f"{self._prefix}/success", 0.0)

            return base.TimeStep(step_type="LAST", reward=reward, discount=0.0, observation=observation)

        # Continue episode
        observation_content = "\n".join(self._conversation_history)
        observation = request.LLMRequest(
            messages=[
                request.UserMessage(role="user", content=observation_content),
            ]
        )

        step_end_time = time.time()
        self._tracker.scalar(f"{self._prefix}/step", step_end_time - step_start_time)

        return base.TimeStep(step_type="MID", reward=reward, discount=1.0, observation=observation)

    async def _get_oracle_answer(self, question: str) -> str:
        """Get yes/no answer from the oracle LLM."""
        if self._hidden_object is None:
            raise RuntimeError("Hidden object not set.")

        # Create oracle prompt
        oracle_prompt = ORACLE_PROMPT_TEMPLATE.format(word=self._hidden_object, question=question)

        oracle_request = request.LLMRequest(
            messages=[
                request.UserMessage(role="user", content=oracle_prompt),
            ],
            temperature=0.0,  # Deterministic answers
        )

        # Call oracle
        with self._tracker.timeit(f"{self._prefix}/oracle_call"):
            oracle_response = await self._oracle_client(oracle_request)

        # Extract answer from response - LLMResponse has data: list[TextData]
        answer_text = oracle_response.data[0].content.strip()

        _LOGGER.debug("[%d] Raw oracle response: %s", id(self), answer_text)

        return answer_text

    def _extract_question_from_response(self, action: response.LLMResponse) -> str:
        """Extract the question text from the agent's response."""
        # Get the text content from the first data element
        question = action.data[0].content if action.data else ""
        return question.strip()

    def _check_if_correct_guess(self, question: str) -> bool:
        """Check if the question is a correct guess of the hidden object."""
        if self._hidden_object is None:
            return False

        question_lower = question.lower()
        object_lower = self._hidden_object.lower()

        # Check if the question contains "is it [object]?" or similar patterns
        # Simple heuristic: check if the object name appears in the question
        return object_lower in question_lower

    async def close(self) -> None:
        """Clean up resources (none needed for this environment)."""
        pass

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        self._is_active = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit async context manager."""
        del exc_type, exc_value, traceback  # Unused
        self._is_active = False
        await self.close()

    def _assert_active(self) -> None:
        """Assert that the environment is active."""
        if not self._is_active:
            raise RuntimeError("Environment is not active. Use 'async with' to activate.")
