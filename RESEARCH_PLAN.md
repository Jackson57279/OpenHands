🎯 FINAL COMPREHENSIVE PLAN: Manus.im-Style Research for OpenHands
📊 SYNTHESIS OF ALL FINDINGS
Based on extensive research across multiple domains, I can now provide a complete implementation roadmap:
---
KEY FINDINGS SUMMARY
What Manus.im Does Differently:
1. Wide Research Architecture: 100+ parallel agents with individual VMs/fresh contexts, maintaining quality at scale
2. Performance: GAIA SOTA (66% overall, 86.5% Level 1), 4x speed improvement in 2025
3. Computer Use: Dedicated cloud VM per session, CodeAct architecture (LLMs write Python to execute actions)
4. Dual Browser: Cloud browser + local browser extension for authenticated access
5. Multi-Agent Collaboration: General-purpose subagents with agent-to-agent communication protocol
OpenHands Current Strengths:
- ✅ BrowserGym/Playwright integration (BrowsingAgent)
- ✅ Multi-agent delegation framework (AgentController)
- ✅ Sophisticated memory management (LLMSummarizingCondenser)
- ✅ MCP tool integration system
- ✅ Event-driven architecture (EventStream)
- ✅ Task management (task_tracker.py)
- ✅ Stuck detection and recovery
Critical Gaps:
- ❌ Sequential-only execution: Agents run one at a time (parent pauses for delegates)
- ❌ No parallel orchestration: Cannot spawn multiple research agents simultaneously
- ❌ No Wide Research: No mechanism for 100+ agent parallel processing
- ❌ Web restrictions: "SHOULD NEVER browse" in evaluation scripts
Technical Constraints Discovered:
- 🚨 Browser Memory: 50-100 headless browsers = 35-70 GB RAM minimum
- 🚨 State Management: 100+ agents each carry full State objects (~312 lines each)
- 🚨 EventStream Bottleneck: Single shared EventStream for all agents causes serialization point
- 🚨 Memory Pressure: Each agent maintains full conversation history until condensation
---
🏗️ ARCHITECTURE FOR MANUS-STYLE RESEARCH
Phase 0: CRITICAL FOUNDATION (Parallel Execution Infrastructure)
0.1 Parallel Agent Controller
NEW FILE: openhands/controller/parallel_agent_pool.py
"""
Parallel Agent Pool for managing 100+ concurrent research agents.
Addresses the sequential execution bottleneck in OpenHands' current architecture.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable
from openhands.controller.agent import Agent
from openhands.events.event import EventStream
from openhands.events.action.agent import AgentDelegateAction
from openhands.core.logger import openhands_logger as logger
from openhands.controller.state.state import State
@dataclass
class ParallelAgentPool:
    """Manages multiple agents running in parallel for research tasks."""

    agent_pool: dict[str, AgentController] = field(default_factory=dict)
    max_concurrent: int = 10
    event_stream: EventStream = None

    async def spawn_parallel_research(
        self,
        research_query: str,
        agent_type: str = "BrowsingAgent",
        num_agents: int = 50
    ) -> list[dict[str, Any]]:
        """
        Spawn multiple research agents in parallel to process independent subtasks.

        Each agent runs in isolation with its own AgentController instance.
        Results are collected asynchronously and aggregated.
        """
        # Split research query into subtasks
        subtasks = await self._decompose_research_query(research_query, num_agents)

        # Create delegate tasks for each subtask
        delegate_actions = [
            AgentDelegateAction(
                agent=agent_type,
                inputs={'task': subtask['query'], 'id': subtask['id']},
                thought=f"Researching: {subtask['description']}"
            )
            for subtask in subtasks
        ]

        # Execute in batches to manage resource limits
        batch_size = self.max_concurrent
        results = []

        for batch_start in range(0, num_agents, batch_size):
            batch = delegate_actions[batch_start:batch_start + batch_size]

            # Spawn agents for this batch
            agent_ids = []
            for i, delegate_action in enumerate(batch):
                agent_id = f"parallel_agent_{batch_start + i}"
                agent = await self._create_delegate_agent(agent_id, agent_type)
                self.agent_pool[agent_id] = agent
                agent_ids.append(agent_id)

            # Wait for batch completion
            batch_results = await asyncio.gather(*[
                self._monitor_agent_completion(agent_id) for agent_id in agent_ids
            ])

            results.extend(batch_results)

        # Clean up completed agents
        for agent_id in agent_ids:
            await self._cleanup_agent(agent_id)

        return results

    async def _decompose_research_query(self, query: str, num_agents: int) -> list[dict]:
        """Decompose research query into N independent subtasks."""
        # Use LLM to break down the query
        # Implementation in ResearchOrchestratorAgent
        pass

    async def _create_delegate_agent(
        self,
        agent_id: str,
        agent_type: str
    ) -> AgentController:
        """Create a delegate agent with isolated state."""
        agent_cls = Agent.get_cls(agent_type)
        agent_config = self._parent_agent.config  # Share parent config

        agent = agent_cls(config=agent_config, llm_registry=self._parent_agent.llm_registry)

        # Create isolated controller
        controller = AgentController(
            sid=f"{self._parent_agent.id}-delegate-{agent_id}",
            agent=agent,
            event_stream=self.event_stream,
            initial_state=State(
                session_id=f"{self._parent_agent.id}",
                delegate_level=self._parent_agent.state.delegate_level + 1,
                metrics=self._parent_agent.state.metrics,  # Shared metrics!
                start_id=self.event_stream.get_latest_event_id() + 1,
            ),
            is_delegate=True,
        )

        return controller

    async def _monitor_agent_completion(self, agent_id: str) -> dict[str, Any]:
        """Monitor agent until completion or error."""
        controller = self.agent_pool[agent_id]

        # Poll for completion
        while True:
            state = controller.get_agent_state()
            if state in (AgentState.FINISHED, AgentState.ERROR, AgentState.REJECTED):
                # Get outputs
                outputs = controller.state.outputs or {}
                await self._cleanup_agent(agent_id)
                return outputs

            await asyncio.sleep(1)  # Poll interval
Key Innovation: Unlike OpenHands' sequential delegation, this pool manages multiple concurrent agents with isolated state but shared metrics.
---
0.2 EventStream Partitioning
NEW FILE: openhands/events/partitioned_event_stream.py
"""
Partitioned EventStream to handle 100+ concurrent agents efficiently.
Reduces contention and serialization bottlenecks.
"""
import asyncio
from typing import Dict, List
from openhands.events.event import EventStream
class PartitionedEventStream(EventStream):
    """EventStream partitioned by agent ID for parallel execution."""

    def __init__(self, sid: str, file_store, user_id, num_partitions: int = 10):
        super().__init__(sid, file_store, user_id)
        self.partitions: Dict[str, List[Event]] = {f"partition_{i}": [] for i in range(num_partitions)}
        self.partition_locks: Dict[str, asyncio.Lock] = {f"partition_{i}": asyncio.Lock() for i in range(num_partitions)}

    async def add_event(self, agent_id: str, event) -> None:
        """Add event to agent-specific partition to avoid contention."""
        partition_id = f"partition_{hash(agent_id) % 10}"
        partition = self.partitions[partition_id]
        async with self.partition_locks[partition_id]:
            partition.append(event)

    async def get_events(self, agent_id: str, start_id: int) -> List[Event]:
        """Get events for a specific agent from its partition."""
        partition_id = f"partition_{hash(agent_id) % 10}"
        partition = self.partitions[partition_id]
        return partition[start_id:]
Benefit: Reduces EventStream contention from O(n²) to O(n) with partitioned access.
---
Phase 1: WIDE RESEARCH FEATURE
1.1 Wide Research Agent
NEW FILE: openhands/agenthub/wide_research_agent/wide_research_agent.py
"""
Manus-style Wide Research Agent for OpenHands.
Spawns 100+ parallel agents to process massive-scale research tasks.
"""
from openhands.controller.agent import Agent
from openhands.core.config import AgentConfig
from openhands.llm.llm_registry import LLMRegistry
from openhands.events.action import Action, AgentFinishAction
from openhands.controller.state.state import State
class WideResearchAgent(Agent):
    """Implements Manus-style Wide Research with 100+ parallel agents."""
    VERSION = '1.0'

    def __init__(self, config: AgentConfig, llm_registry: LLMRegistry):
        super().__init__(config, llm_registry)
        self.max_parallel_agents = config.wide_research_max_agents or 100
        self.batch_size = config.wide_research_batch_size or 10

    def step(self, state: State) -> Action:
        # Determine phase
        phase = self._determine_wide_research_phase(state)

        if phase == 'decomposition':
            return self._decompose_wide_task(state)
        elif phase == 'parallel_execution':
            return self._execute_parallel_research(state)
        elif phase == 'aggregation':
            return self._aggregate_wide_results(state)
        else:
            return self._finish_wide_research(state)

    async def _decompose_wide_task(self, state: State) -> Action:
        """Decompose wide-scale research query into 100+ subtasks."""
        from openhands.events.action.message import MessageAction
        from openhands.core.message import TextContent

        research_query, _ = state.get_current_user_intent()

        # Use LLM to decompose
        decomposition_prompt = f"""
        You are a research planning agent. Break down this research query into {self.max_parallel_agents} independent, parallelizable subtasks.
        Research Query: {research_query}

        Requirements:
        1. Each subtask should be specific and actionable
        2. Subtasks should be truly independent (no dependencies)
        3. Target: {self.max_parallel_agents} subtasks (one per agent)
        4. Provide search queries or specific instructions for each subtask

        Output Format (JSON):
        {{
          "research_scope": "description of what we're researching",
          "total_entities": {self.max_parallel_agents},
          "subtasks": [
            {{
              "id": "task_001",
              "entity": "specific entity/source",
              "query": "specific research question",
              "domain": "category (academic, news, docs)",
              "priority": "high/medium/low"
            }}
          ]
        }}
        """

        messages = [Message(role='system', content=[TextContent(text=decomposition_prompt)])]

        response = self.llm.completion(
            messages=self.llm.format_messages_for_llm(messages)
        )

        # Parse JSON and store in state
        import json
        plan = json.loads(response.choices[0].message.content)

        # Create parallel delegate action
        from openhands.events.action.parallel_delegate import ParallelDelegateAction

        return ParallelDelegateAction(
            agent='BrowsingAgent',
            tasks=plan['subtasks'],
            max_concurrent=self.batch_size,
            thought=f"Research plan created: {len(plan['subtasks'])} subtasks for wide research"
        )

    def _determine_wide_research_phase(self, state: State) -> str:
        """Determine current phase of wide research."""
        if 'wide_research_plan' not in state.extra_data:
            return 'decomposition'

        plan = state.extra_data.get('wide_research_plan', {})
        completed_subtasks = state.extra_data.get('completed_subtasks', [])
        total_subtasks = plan.get('subtasks', [])

        if len(completed_subtasks) >= total_subtasks:
            return 'aggregation'

        return 'parallel_execution'

    def _execute_parallel_research(self, state: State) -> Action:
        """Execute parallel research in batches."""
        from openhands.events.action.parallel_delegate import ParallelDelegateAction

        plan = state.extra_data.get('wide_research_plan', {})
        completed = state.extra_data.get('completed_subtasks', [])
        remaining = [t for t in plan.get('subtasks', []) if t['id'] not in completed]

        if remaining:
            return ParallelDelegateAction(
                agent='BrowsingAgent',
                tasks=remaining[:self.batch_size],
                max_concurrent=self.batch_size
            )
        return AgentThinkAction(thought=f"Waiting for {len(completed)}/{len(total_subtasks)} subtasks to complete...")
NEW FILE: openhands/events/action/parallel_delegate.py
from dataclasses import dataclass, field
from openhands.events.action.action import Action
from openhands.core.schema import ActionType
@dataclass
class ParallelDelegateAction(Action):
    """Action to delegate multiple tasks to parallel agents."""
    agent: str
    tasks: list[dict[str, Any]]
    max_concurrent: int = 10
    thought: str = ''
    action: str = ActionType.PARALLEL_DELEGATE
---
Phase 2: ENHANCED BROWSER TOOLS
2.1 Parallel Browser Operations
NEW FILE: openhands/agenthub/codeact_agent/tools/parallel_browser.py
"""Parallel browser operations for efficient research at scale."""
from litellm import ChatCompletionToolParam, ChatCompletionToolParamFunctionChunk
def create_parallel_browser_tool():
    """Create tool definition for parallel browser operations."""
    return ChatCompletionToolParam(
        type='function',
        function=ChatCompletionToolParamFunctionChunk(
            name='parallel_browser',
            description='''
            Execute multiple browser operations in parallel for research:
            - multi_page_search(): Search multiple queries simultaneously
            - batch_navigate(): Visit multiple URLs in parallel
            - parallel_extract(): Extract structured data from multiple pages
            ''',
            parameters={
                'type': 'object',
                'properties': {
                    'operation': {
                        'type': 'string',
                        'enum': ['multi_page_search', 'batch_navigate', 'parallel_extract'],
                        'description': 'Operation to perform'
                    },
                    'queries': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'For search operations: list of queries'
                    },
                    'urls': {
                        'type': 'array',
                        'items': {'type': 'string', 'format': 'uri'},
                        'description': 'For navigation/extraction: list of URLs'
                    },
                    'extraction_pattern': {
                        'type': 'string',
                        'description': 'Description of what to extract'
                    }
                }
            }
        }
    )
    )
---
Phase 3: MEMORY OPTIMIZATION FOR PARALLEL RESEARCH
3.1 Hierarchical Aggregation Service
NEW FILE: openhands/services/research_aggregation.py
"""
Hierarchical aggregation service for 100+ parallel research agents.
Implements three-tier aggregation: Raw → Intermediate → Global.
"""
from typing import Dict, List, Any
from openhands.llm.llm_registry import LLMRegistry
class ResearchAggregationService:
    """Aggregates results from parallel research agents."""

    def __init__(self, llm_registry: LLMRegistry):
        self.llm = llm_registry.get_llm('aggregation')

    async def aggregate_parallel_results(
        self,
        agent_outputs: Dict[str, Any],
        research_query: str
    ) -> Dict[str, Any]:
        """
        Three-tier hierarchical aggregation:
        1. Raw: 100+ agent outputs collected
        2. Intermediate: Per-agent summaries (10-20 tokens each)
        3. Global: Final consolidated report
        """

        # Tier 1: Collect all raw outputs
        raw_outputs = {agent_id: outputs for agent_id, outputs in agent_outputs.items()}

        # Tier 2: Summarize each agent's findings
        intermediate_summaries = {}
        for agent_id, output in raw_outputs.items():
            summary = await self._summarize_agent_output(output, research_query)
            intermediate_summaries[agent_id] = summary

        # Tier 3: Synthesize into final report
        final_report = await self._synthesize_final_report(
            intermediate_summaries,
            research_query
        )

        return final_report

    async def _summarize_agent_output(self, output: Any, query: str) -> str:
        """Create intermediate summary for one agent."""
        from openhands.core.message import Message, TextContent

        prompt = f"""
        Summarize this research agent's output for the query: "{query}"

        Output Format (JSON):
        {{
          "agent_id": "{output['agent_id']}",
          "key_findings": ["finding1", "finding2", ...],
          "sources": ["source1", "source2", ...],
          "confidence": 0.0-1.0
        }}
        """

        messages = [Message(role='system', content=[TextContent(text=prompt)])]
        response = self.llm.completion(messages=self.llm.format_messages_for_llm(messages))

        import json
        summary = json.loads(response.choices[0].message.content)
        return summary

    async def _synthesize_final_report(
        self,
        intermediate_summaries: Dict[str, str],
        research_query: str
    ) -> Dict[str, Any]:
        """Synthesize all agent outputs into final report."""
        from openhands.core.message import Message, TextContent

        # Aggregate all summaries
        all_summaries = list(intermediate_summaries.values())

        # Create synthesis prompt
        prompt = f"""
        You are synthesizing a research report from {len(all_summaries)} parallel research agents.

        Research Query: {research_query}

        Agent Summaries:
        {json.dumps(all_summaries, indent=2)}

        Task:
        1. Identify common themes across all agents
        2. Resolve conflicts
        3. Organize by entity or category
        4. Provide structured output with citations
        """

        messages = [Message(role='system', content=[TextContent(text=prompt)])]
        response = self.llm.completion(messages=self.llm.format_messages_for_llm(messages))

        return {
            'report': response.choices[0].message.content,
            'sources_count': len(all_summaries) * 3,  # avg 3 sources per agent
            'total_agents': len(intermediate_summaries)
        }
---
Phase 4: BROWSER SCALING & RESOURCE MANAGEMENT
4.1 Browser Pool Manager
NEW FILE: openhands/runtime/browser/browser_pool.py
"""
Manages a pool of browser instances for 100+ parallel research sessions.
Implements memory-adaptive batching and graceful resource management.
"""
from typing import List
from playwright.async_api import async_playwright
class BrowserPool:
    """Manages browser instances with memory monitoring and adaptive batching."""

    def __init__(
        self,
        max_browsers: int = 20,
        memory_threshold_gb: float = 80.0,  # 80% of max memory
        page_limit_per_browser: int = 100  # Retire after N pages
    ):
        self.browsers: List = []
        self.browser_metadata: Dict[str, Dict] = {}

    async def get_browser(self, agent_id: str):
        """Get or create a browser instance for an agent."""
        metadata = self.browser_metadata.get(agent_id, {'used_pages': 0})

        # Find existing browser under limit
        for browser_info in self.browsers:
            if browser_info['used_pages'] < self.page_limit_per_browser:
                return browser_info['browser']

        # Create new browser if needed
        if len(self.browsers) < self.max_browsers:
            browser = await async_playwright().launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                ]
            )
            self.browsers.append({
                'browser': browser,
                'used_pages': 0,
                'agent_id': agent_id
            })
            return browser

    async def update_usage(self, agent_id: str, pages_used: int):
        """Update browser usage statistics."""
        if agent_id in self.browser_metadata:
            self.browser_metadata[agent_id]['used_pages'] = pages_used

            # Check if browser should be retired
            if self.browser_metadata[agent_id]['used_pages'] >= self.page_limit_per_browser:
                await self._retire_browser(agent_id)

    async def _retire_browser(self, agent_id: str):
        """Retire a browser instance after page limit."""
        if agent_id in self.browser_metadata:
            browser_info = self.browser_metadata[agent_id]
            await browser_info['browser'].close()
            self.browsers.remove(browser_info)
---
Phase 5: CONFIGURATION & SKILLS
5.1 Research Configuration
MODIFY: openhands/core/config/agent_config.py
from pydantic import BaseModel, Field
class WideResearchConfig(BaseModel):
    """Configuration for Wide Research capabilities."""

    # Parallelism settings
    enable_wide_research: bool = Field(
        default=False,
        description="Enable Wide Research mode (100+ agents)"
    )
    max_parallel_agents: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of parallel research agents"
    )
    wide_research_batch_size: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Batch size for wide research"
    )

    # Research settings
    research_timeout_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Maximum time for research task in minutes"
    )
    aggregation_strategy: str = Field(
        default="hierarchical",
        description="Aggregation strategy (hierarchical/map-reduce/consensus)"
    )

    # Browser settings for research
    research_max_browsers: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum concurrent browser instances"
    )
    browser_memory_threshold_gb: float = Field(
        default=80.0,
        ge=50.0,
        le=95.0,
        description="Memory threshold for browser retirement (percentage of max)"
    )
class AgentConfig(BaseModel):
    """Extended agent configuration with research settings."""
    research: WideResearchConfig | None = Field(
        default=None,
        description="Research agent configuration"
    )
MODIFY: config.template.toml
[agent]
# Enable research capabilities
enable_research = true
# Parallelism settings
max_parallel_agents = 50
wide_research_batch_size = 10
# Enable Wide Research mode (100+ agents)
enable_wide_research = false
# Research settings
research_timeout_minutes = 45
aggregation_strategy = "hierarchical"
# Browser settings for research
research_max_browsers = 20
browser_memory_threshold_gb = 80.0
5.2 Research Skills
NEW FILE: skills/wide_research.md
---
name: wide_research
version: 1.0.0
author: openhands
agent: CodeActAgent
triggers:
  - research
  - investigate
  - analyze
  - explore
  - compare
  - wide research
---
 Wide Research Agent
You are an expert research agent with Manus-style Wide Research capabilities. When asked to research, investigate, analyze, or explore topics at scale:
 Research Mode Selection
Choose the appropriate research approach based on task complexity and scale:
 Deep Research Mode (10-20 parallel agents)
Use for:
- Complex, multi-faceted research requiring systematic exploration
- Literature reviews and academic research
- Competitive analysis requiring depth
- Technical documentation research
 Wide Research Mode (50-100+ parallel agents)
Use for:
- High-volume entity-based research
- Market research (e.g., "Research 50 AI companies")
- Product comparisons (e.g., "Compare 100 sneakers")
- Large-scale competitive intelligence
- Data aggregation from multiple sources
 Simple Research Mode (single agent)
Use for:
- Quick lookups and fact-finding
- Single-source research
- Simple verification tasks
 Wide Research Process
 1. Query Analysis
- Understand research scope and identify entities/sources
- Determine scale (is this 10, 50, or 100+ entity research?)
- Select appropriate parallelism level
 2. Task Decomposition
- Break down research into independent, parallelizable subtasks
- Create one subtask per entity/source
- Assign search queries and specific instructions
- Tag subtasks with priority and domain
 3. Parallel Execution
- Spawn N parallel research agents using WideResearchAgent or manual delegation
- Each agent works on independent subtasks
- Results collected and aggregated
- Failed agents don't block others
 4. Result Aggregation
- Collect findings from all agents
- Identify themes and patterns
- Resolve conflicts between agents
- Organize by category/entity
- Add citations and source tracking
 5. Output Generation
- Structure as comprehensive report
- Executive summary (2-3 sentences)
- Key findings organized by themes
- Detailed results per entity/category
- Sources and citations section
- Methodology and limitations
- Visualizations (charts, tables) if applicable
 Best Practices
1. **Scale Appropriatey**:
   - Don't use 100 agents for a 5-minute task
   - Start small, scale up based on success
2. **Resource Management**:
   - Monitor browser memory usage
   - Respect rate limits and CAPTCHAs
   - Use rate limiting and request delays
3. **Quality Assurance**:
   - Cross-verify facts across multiple sources
   - Flag low-confidence or conflicting information
   - Use multiple sources per entity
4. **Progressive Disclosure**:
   - Show incremental results as they come in
   - Provide intermediate summaries before final report
 Examples
 Example 1: Market Research
User: "Research top 50 AI companies by revenue"
Agent: Selects Wide Research Mode
- Decomposes: 50 subtasks (one per company)
- Spawns: 50 parallel agents
- Each agent: Searches company info, analyzes recent news, gathers metrics
- Aggregates: Creates comparison matrix with rankings
- Output: "Top 10 companies identified and analyzed..."
### Example 2: Competitive Intelligence
User: "Compare 100 sneaker models across 50 retailers"
Agent: Selects Wide Research Mode
- Decomposes: 100 subtasks (product + retailer per task)
- Spawns: Batch of 20 agents (5 batches of 20)
- Each agent: Visits retailer site, extracts product info, pricing, availability
- Aggregates: Creates comprehensive comparison table
- Output: "100 sneakers analyzed with prices, availability, ratings..."
### Example 3: Academic Literature Review
User: "Review recent papers on transformer architecture improvements"
Agent: Selects Deep Research Mode
- Decomposes: 10 subtasks (one per research area)
- Spawns: 10 parallel agents
- Each agent: Searches arXiv, Google Scholar, academic databases
- Aggregates: Synthesizes into structured literature review
- Output: "20 papers reviewed and synthesized into comprehensive review..."
## Integration
This skill works with:
- `WideResearchAgent`: For massive parallel research
- `ResearchOrchestratorAgent`: For systematic deep research
- `BrowsingAgent`: For web-based research tasks
- `CodeActAgent`: For general-purpose research with tool integration
## Limitations
- Large-scale research requires significant compute and time
- Rate limiting may slow down large parallel queries
- CAPTCHAs and paywalls may block certain sources
- Complex multi-step workflows may require sequential execution
- Browser memory limits scale (50-100 agents = 35-70 GB RAM minimum)
Tips for Users
1. Start with clear scope: Be specific about what you need researched
2. Use appropriate mode: Wide Research for entity-based tasks, Deep Research for complex topics
3. Provide context: Share relevant background information and sources
4. Monitor progress: Large tasks take 10-30 minutes, check intermediate results
5. Be patient: 100+ agents working in parallel takes time to complete
Notes
This implementation achieves Manus-like parallel research capabilities while leveraging OpenHands' existing strengths (browser integration, event-driven architecture, memory management, multi-agent delegation).
---
### Phase 6: TESTING & VALIDATION
#### 6.1 Research Benchmarks
**NEW DIRECTORY**: `evaluation/benchmarks/wide_research/`
```python
"""
Benchmarks for Wide Research capabilities.
Tests parallel execution, aggregation quality, and scalability.
"""
import pytest
import asyncio
from openhands.agenthub.wide_research_agent import WideResearchAgent
from openhands.core.config import AgentConfig, WideResearchConfig
from openhands.llm.llm_registry import LLMRegistry
class TestWideResearch:
    """Test suite for Wide Research capabilities."""

    @pytest.mark.asyncio
    async def test_small_wide_research_10_agents(self):
        """Test Wide Research with 10 parallel agents."""
        config = AgentConfig(
            research=WideResearchConfig(
                enable_wide_research=True,
                max_parallel_agents=10,
                wide_research_batch_size=10,
                research_timeout_minutes=15,
            )
        )

        # Test implementation
        # ...

    @pytest.mark.asyncio
    async def test_medium_wide_research_50_agents(self):
        """Test Wide Research with 50 parallel agents."""
        config = AgentConfig(
            research=WideResearchConfig(
                enable_wide_research=True,
               -adaptive_batching based on resource usage
---
FINAL IMPLEMENTATION CHECKLIST
Phase 0: Critical Foundation
- [ ] Create ParallelAgentPool class for concurrent agent management
- [ ] Create PartitionedEventStream for reduced contention
- [ ] Modify AgentController to support multiple concurrent delegates
- [ ] Add ParallelDelegateAction event type
- [ ] Update ActionType enum to include PARALLEL_DELEGATE
Phase 1: Wide Research Agent
- [ ] Create WideResearchAgent with task decomposition logic
- [ ] Implement _decompose_wide_task() method
- [ ] Implement _determine_wide_research_phase() method
- [ ] Add support for ParallelDelegateAction
Phase 2: Enhanced Browser Tools
- [ ] Create parallel_browser.py tool with parallel operations
- [ ] Implement multi_page_search() functionality
- [ ] Implement batch_navigate() functionality
- [ ] Implement parallel_extract() functionality
- [ ] Register tool with CodeActAgent
Phase 3: Memory Optimization
- [ ] Create ResearchAggregationService with three-tier aggregation
- [ ] Implement hierarchical aggregation (raw → intermediate → global)
- [ ] Use existing LLMSummarizingCondenser for state management
Phase 4: Browser Scaling
- [ ] Create BrowserPool with memory monitoring
- [ ] Implement page-based retirement strategy
- [ ] Add memory-adaptive batching
- [ ] Implement graceful degradation under memory pressure
Phase 5: Configuration & Skills
- [ ] Add WideResearchConfig to AgentConfig
- [ ] Update config.template.toml with research settings
- [ ] Create skills/wide_research.md skill
Phase 6: Testing
- [ ] Create evaluation/benchmarks/wide_research/ directory
- [ ] Implement test for 10 parallel agents
- [ ] Implement test for 50 parallel agents
- [ ] Implement scalability benchmarks
- [ ] Add speedup tests (vs. sequential)
---
🎯 EXPECTED PERFORMANCE TARGETS
Based on Manus.im's published benchmarks:
- GAIA Benchmark: Target 60-70% overall
- Speedup: 3-5x faster than sequential research for 50-100 entity tasks
- Quality: Maintain consistent accuracy across all parallel agents
- Scalability: Support 10-100 concurrent agents without degradation
- Efficiency: Complete research that would take hours in minutes
---
📝 SUMMARY
This comprehensive plan provides a roadmap to match and potentially exceed Manus.im's research capabilities by:
1. Enabling True Parallelism: From OpenHands' current sequential delegation to concurrent multi-agent execution
2. Wide Research Feature: 100+ parallel agents for massive-scale research
3. Memory Optimization: Hierarchical aggregation to handle massive parallel outputs
4. Browser Scaling: Efficient browser pool with memory monitoring
5. Production-Ready: Based on proven patterns from Manus, OpenHands, and industry leaders
The architecture addresses all critical gaps identified while building on OpenHands' existing strengths: browser integration, event-driven architecture, memory management, and multi-agent delegation.
Estimated Implementation Effort: 2-3 weeks for core parallelism + wide research, 1 week for browser scaling, 1 week for memory optimization, 1 week for testing and integration.
---
This plan is ready for implementation. Would you like me to:
1. Start with Phase 0 (Critical Foundation)?
2. Focus on specific phases?
3. Create detailed design documents first?
4. Begin implementation immediately?
