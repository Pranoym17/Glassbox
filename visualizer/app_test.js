import {
  buildGraph,
  firstSequence,
  graphAssertions,
  validateEvents,
} from "./app.js";

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function fixture(name) {
  return JSON.parse(
    await Deno.readTextFile(new URL(`fixtures/${name}`, import.meta.url)),
  );
}

function distinctIds(events) {
  const ids = new Set();
  events.forEach((event) => {
    ids.add(event.output_id);
    event.input_ids.forEach((id) => ids.add(id));
  });
  return ids;
}

Deno.test("tiny fixture is a complete acyclic step", async () => {
  const events = await fixture("tiny_step.json");
  assert(events.length === 84, "tiny event count changed");
  assert(validateEvents(events) === 0, "unexpected step");
  const graph = buildGraph(events);
  assert(graph.nodes.length === 53, "tiny distinct node count changed");
  assert(
    graph.nodes.length === distinctIds(events).size,
    "node count is not distinct",
  );
  assert(
    graph.edges.every((edge) =>
      graph.byId.has(edge.source) && graph.byId.has(edge.target)
    ),
    "missing edge endpoint",
  );
  assert(
    graph.edges.every((edge) =>
      graph.byId.get(edge.source).depth < graph.byId.get(edge.target).depth
    ),
    "graph is not acyclic",
  );
  assert(graph.edges.length === 56, "tiny edge count changed");
  assert(graphAssertions(events, graph), "graph assertions failed");
});

Deno.test("full fixture filters to exactly one sequence graph", async () => {
  const events = await fixture("step.json");
  assert(events.length === 902, "full event count changed");
  validateEvents(events);
  const selected = firstSequence(events);
  assert(
    selected.filter((event) =>
      event.phase === "forward" && event.op === "cross_entropy"
    ).length === 1,
    "sequence filter kept duplicate losses",
  );
  const graph = buildGraph(selected);
  assert(graph.nodes.length === 91, "full filtered node count changed");
  assert(
    graph.nodes.length === distinctIds(selected).size,
    "filtered node count is not distinct",
  );
  assert(graph.edges.length === 98, "full filtered edge count changed");
  assert(graphAssertions(selected, graph), "filtered graph assertions failed");
});

Deno.test("cycle detection fails loudly", () => {
  const events = [
    {
      step: 0,
      phase: "forward",
      op: "a",
      output_id: 1,
      input_ids: [2],
      shape: [1],
    },
    {
      step: 0,
      phase: "forward",
      op: "b",
      output_id: 2,
      input_ids: [1],
      shape: [1],
    },
  ];
  let failed = false;
  try {
    buildGraph(events);
  } catch (error) {
    failed = error.message === "graph is cyclic";
  }
  assert(failed, "cycle did not produce the expected failure");
});
