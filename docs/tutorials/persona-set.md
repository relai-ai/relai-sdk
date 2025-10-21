<h1>Persona Set</h1>

An AI persona can be used to mimic a particular role by defining a system prompt and optionally equipping it with tools. 
A persona set is set of personas, which can be conveniently curated and/or imported through RELAI platform.

<h2>Create Persona Set</h2>

1. To create a persona set, first go to RELAI platform and find [Persona Sets](https://platform.relai.ai/agent-hub/persona) under AgentHub.

    <img src="../assets/tutorials/persona-set/1.png" alt="RELAI platform->AgentHub->Persona Sets"/>

2. Click "Create Persona Set".

    <img src="../assets/tutorials/persona-set/2.png" alt="Create Persona Set"/>

3. Name the new persona set; Upload a CSV file with a `system_prompts` column to populate the list, or add prompts manually.

    <img src="../assets/tutorials/persona-set/3.png" alt="Upload a CSV file with a `system_prompts` column to populate the list, or add prompts manually."/>

4. And done! Your persona set is created. You can chat with personas directly here and adjust them further whenever needed.

    <img src="../assets/tutorials/persona-set/4.png" alt="Successful persona set creation"/>

<h2>Use Persona Set in Simulation</h2>

1. Decorate inputs/tools that will be simulated.

    ```python
    from relai import simulated

    @simulated
    async def get_user_input():
        msg = input("User: ")
        return msg
    ```

2. When setting up the simulation environment, bind the persona set to the corresponding fully-qualified function names.

    ```python
    from relai.simulator import AsyncSimulator, random_env_generator

    env_generator = random_env_generator(
        {"__main__.get_user_input": PersonaSet(persona_set_id="your_persona_set_id_here")}
    )

    async def main():
        async with AsyncRELAI() as client:
            simulator = AsyncSimulator(
                client=client,
                agent_fn=<your agent function here>, 
                env_generator=env_generator,
                log_runs=True,
            )

            agent_logs = await simulator.run(num_runs=4)
            print(agent_logs)

    asyncio.run(main())
    ```

