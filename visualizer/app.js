export function validateEvents(events) {
  if (!Array.isArray(events) || events.length === 0) {
    throw new Error("step has no events");
  }
  const fields = ["step", "phase", "op", "output_id", "input_ids", "shape"];
  const step = events[0].step;
  let backward = false;
  for (const [index, event] of events.entries()) {
    for (const field of fields) {
      if (!(field in event)) {
        throw new Error(`event ${index} is missing ${field}`);
      }
    }
    if (event.step !== step) {
      throw new Error("fixture contains more than one step");
    }
    if (!["forward", "backward"].includes(event.phase)) {
      throw new Error(`event ${index} has invalid phase`);
    }
    if (!Array.isArray(event.input_ids) || !Array.isArray(event.shape)) {
      throw new Error(`event ${index} has invalid arrays`);
    }
    if (event.phase === "backward") {
      backward = true;
      if (
        typeof event.grad_norm !== "number" || !Number.isFinite(event.grad_norm)
      ) {
        throw new Error(`backward event ${index} has invalid grad_norm`);
      }
    } else if (backward) {
      throw new Error("forward event appears after backward traversal began");
    }
  }
  if (!backward) throw new Error("step has no backward traversal");
  return step;
}

export function firstSequence(events) {
  const selected = new Set();
  let complete = false;
  return events.filter((event) => {
    if (event.phase === "forward") {
      if (complete) return false;
      selected.add(event.output_id);
      event.input_ids.forEach((id) => selected.add(id));
      if (event.op === "cross_entropy") complete = true;
      return true;
    }
    return selected.has(event.output_id);
  });
}

export function buildGraph(events) {
  const nodes = new Map();
  const edges = new Map();
  const ids = new Set();
  const ensure = (id, shape = [], op = "leaf") => {
    ids.add(id);
    if (!nodes.has(id)) nodes.set(id, { id, shape, op, depth: 0 });
    const node = nodes.get(id);
    if (shape.length) node.shape = shape;
    if (op) node.op = op;
  };

  for (const event of events) {
    ensure(event.output_id, event.shape, event.op);
    event.input_ids.forEach((id) => ensure(id));
    if (event.phase !== "forward") continue;
    for (const input of event.input_ids) {
      if (input === event.output_id) {
        throw new Error(`self edge at tensor ${input}`);
      }
      const key = `${input}->${event.output_id}`;
      if (!edges.has(key)) {
        edges.set(key, {
          key,
          source: input,
          target: event.output_id,
          op: event.op,
        });
      }
    }
  }

  if (nodes.size !== ids.size) {
    throw new Error("distinct tensor count mismatch");
  }
  for (const edge of edges.values()) {
    if (!nodes.has(edge.source) || !nodes.has(edge.target)) {
      throw new Error(`edge ${edge.key} has a missing endpoint`);
    }
  }

  const indegree = new Map([...nodes.keys()].map((id) => [id, 0]));
  const outgoing = new Map([...nodes.keys()].map((id) => [id, []]));
  for (const edge of edges.values()) {
    indegree.set(edge.target, indegree.get(edge.target) + 1);
    outgoing.get(edge.source).push(edge.target);
  }
  const queue = [...nodes.keys()].filter((id) => indegree.get(id) === 0).sort((
    a,
    b,
  ) => a - b);
  let visited = 0;
  while (queue.length) {
    const id = queue.shift();
    visited += 1;
    for (const target of outgoing.get(id)) {
      nodes.get(target).depth = Math.max(
        nodes.get(target).depth,
        nodes.get(id).depth + 1,
      );
      indegree.set(target, indegree.get(target) - 1);
      if (indegree.get(target) === 0) {
        queue.push(target);
        queue.sort((a, b) => a - b);
      }
    }
  }
  if (visited !== nodes.size) throw new Error("graph is cyclic");

  const layers = new Map();
  for (const node of nodes.values()) {
    if (!layers.has(node.depth)) layers.set(node.depth, []);
    layers.get(node.depth).push(node);
  }
  let largestLayer = 1;
  for (const layer of layers.values()) {
    layer.sort((a, b) => a.id - b.id);
    largestLayer = Math.max(largestLayer, layer.length);
    layer.forEach((node, index) => {
      node.x = 110 + node.depth * 210;
      node.y = 80 + index * 68;
    });
  }
  const values = [...nodes.values()];
  return {
    nodes: values,
    byId: nodes,
    edges: [...edges.values()],
    maxDepth: Math.max(...values.map((node) => node.depth), 0),
    largestLayer,
  };
}

export function graphAssertions(events, graph) {
  const ids = new Set();
  events.forEach((event) => {
    ids.add(event.output_id);
    event.input_ids.forEach((id) => ids.add(id));
  });
  if (graph.nodes.length !== ids.size) {
    throw new Error("distinct node assertion failed");
  }
  if (
    graph.edges.some((edge) =>
      !graph.byId.has(edge.source) || !graph.byId.has(edge.target)
    )
  ) {
    throw new Error("edge endpoint assertion failed");
  }
  if (
    graph.edges.some((edge) =>
      graph.byId.get(edge.source).depth >= graph.byId.get(edge.target).depth
    )
  ) {
    throw new Error("topological depth assertion failed");
  }
  return true;
}

function startBrowser() {
  const d3 = globalThis.d3;
  const elements = Object.fromEntries(
    [
      "banner",
      "connection",
      "edge-count",
      "empty",
      "event-count",
      "file",
      "first-sequence",
      "grad-max",
      "grad-min",
      "graph",
      "load-full",
      "load-tiny",
      "next",
      "node-count",
      "play",
      "previous",
      "scrub",
      "source",
      "speed",
      "step",
    ].map((id) => [id, document.getElementById(id)]),
  );
  const state = {
    raw: [],
    events: [],
    graph: null,
    index: 0,
    timer: null,
    label: "",
    nodeSelection: null,
    forwardSelection: null,
    backwardLayer: null,
    widthScale: null,
    colorScale: null,
  };
  const nodeWidth = 142;

  function showError(error) {
    pause();
    elements.banner.textContent = error instanceof Error
      ? error.message
      : String(error);
    elements.banner.hidden = false;
  }

  function clearError() {
    elements.banner.hidden = true;
  }

  function path(edge, reverse = false) {
    const source = state.graph.byId.get(reverse ? edge.target : edge.source);
    const target = state.graph.byId.get(reverse ? edge.source : edge.target);
    const start = source.x + (reverse ? -nodeWidth / 2 : nodeWidth / 2);
    const end = target.x + (reverse ? nodeWidth / 2 : -nodeWidth / 2);
    const middle = (start + end) / 2;
    return `M${start},${source.y} C${middle},${source.y} ${middle},${target.y} ${end},${target.y}`;
  }

  function renderGraph() {
    const svg = d3.select(elements.graph);
    svg.selectAll("*").remove();
    const width = Math.max(1200, state.graph.maxDepth * 210 + 320);
    const height = Math.max(720, state.graph.largestLayer * 68 + 140);
    svg.attr("viewBox", `0 0 ${width} ${height}`);

    const definitions = svg.append("defs");
    definitions
      .append("marker")
      .attr("id", "arrow-forward")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 8)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#3b82f6");
    definitions
      .append("marker")
      .attr("id", "arrow-backward")
      .attr("viewBox", "0 -5 10 10")
      .attr("refX", 8)
      .attr("markerWidth", 6)
      .attr("markerHeight", 6)
      .attr("orient", "auto")
      .append("path")
      .attr("d", "M0,-5L10,0L0,5")
      .attr("fill", "#fb923c");

    const scene = svg.append("g");
    state.forwardSelection = scene
      .append("g")
      .selectAll("path")
      .data(state.graph.edges, (edge) => edge.key)
      .join("path")
      .attr("class", "edge")
      .attr("d", (edge) => path(edge))
      .attr("marker-end", "url(#arrow-forward)");

    scene
      .append("g")
      .selectAll("text")
      .data(state.graph.edges, (edge) => edge.key)
      .join("text")
      .attr("class", "edge-label")
      .attr(
        "x",
        (edge) =>
          (state.graph.byId.get(edge.source).x +
            state.graph.byId.get(edge.target).x) / 2,
      )
      .attr(
        "y",
        (edge) =>
          (state.graph.byId.get(edge.source).y +
              state.graph.byId.get(edge.target).y) / 2 - 5,
      )
      .text((edge) => edge.op);

    state.backwardLayer = scene.append("g");
    state.nodeSelection = scene
      .append("g")
      .selectAll("g")
      .data(state.graph.nodes, (node) => node.id)
      .join("g")
      .attr("class", "node")
      .attr("transform", (node) => `translate(${node.x},${node.y})`);
    state.nodeSelection
      .append("rect")
      .attr("x", -nodeWidth / 2)
      .attr("y", -23)
      .attr("width", nodeWidth)
      .attr("height", 46)
      .attr("rx", 6);
    state.nodeSelection
      .append("text")
      .attr("x", -nodeWidth / 2 + 8)
      .attr("y", -4)
      .text((node) => `#${node.id} ${node.op}`);
    state.nodeSelection
      .append("text")
      .attr("class", "shape")
      .attr("x", -nodeWidth / 2 + 8)
      .attr("y", 13)
      .text((node) =>
        node.shape.length ? `[${node.shape.join("x")}]` : "shape pending"
      );

    svg.call(
      d3.zoom().scaleExtent([0.15, 4]).on(
        "zoom",
        (event) => scene.attr("transform", event.transform),
      ),
    );

    const norms = state.events
      .filter((event) => event.phase === "backward" && event.grad_norm > 0)
      .map((event) => event.grad_norm);
    let minimum = Math.min(...norms);
    let maximum = Math.max(...norms);
    if (!Number.isFinite(minimum)) [minimum, maximum] = [1e-12, 1];
    if (minimum === maximum) maximum = minimum * 10;
    state.widthScale = d3.scaleLog().domain([minimum, maximum]).range([1.25, 8])
      .clamp(true);
    const normalized = d3.scaleLog().domain([minimum, maximum]).range([0, 1])
      .clamp(true);
    state.colorScale = (value) =>
      d3.interpolateRgb("#7c2d12", "#fdba74")(
        normalized(Math.max(value, minimum)),
      );
    elements["grad-min"].textContent = minimum.toExponential(2);
    elements["grad-max"].textContent = maximum.toExponential(2);
  }

  function renderFrame() {
    const active = state.events.slice(0, state.index + 1);
    const forward = new Set();
    const backward = new Map();
    for (const event of active) {
      if (event.phase === "forward") forward.add(event.output_id);
      else backward.set(event.output_id, event);
    }

    state.nodeSelection
      .attr(
        "class",
        (node) =>
          `node ${
            backward.has(node.id)
              ? "backward"
              : forward.has(node.id)
              ? "forward"
              : ""
          }`,
      )
      .select("rect")
      .style("fill", (node) => {
        const event = backward.get(node.id);
        return event
          ? state.colorScale(Math.max(event.grad_norm, 1e-12))
          : null;
      });

    state.forwardSelection.style(
      "opacity",
      (edge) => forward.has(edge.target) ? 0.75 : 0.12,
    );
    const backwardEdges = state.graph.edges
      .filter((edge) => backward.has(edge.target))
      .map((edge) => ({ ...edge, norm: backward.get(edge.target).grad_norm }));
    state.backwardLayer
      .selectAll("path")
      .data(backwardEdges, (edge) => edge.key)
      .join("path")
      .attr("class", "backward-edge")
      .attr("d", (edge) => path(edge, true))
      .attr("marker-end", "url(#arrow-backward)")
      .attr("stroke", (edge) => state.colorScale(Math.max(edge.norm, 1e-12)))
      .attr(
        "stroke-width",
        (edge) => state.widthScale(Math.max(edge.norm, 1e-12)),
      )
      .attr("opacity", 0.82);

    elements.scrub.value = state.index;
    elements["event-count"].textContent = `${
      state.index + 1
    } / ${state.events.length}`;
    elements.previous.disabled = state.index === 0;
    elements.next.disabled = state.index + 1 >= state.events.length;
  }

  function load(events, label) {
    clearError();
    pause();
    validateEvents(events);
    state.raw = events;
    state.events = elements["first-sequence"].checked
      ? firstSequence(events)
      : events.slice();
    state.graph = buildGraph(state.events);
    graphAssertions(state.events, state.graph);
    state.label = label;
    state.index = 0;
    elements.source.textContent = label;
    elements.step.textContent = state.events[0].step;
    elements["node-count"].textContent = state.graph.nodes.length;
    elements["edge-count"].textContent = state.graph.edges.length;
    elements.scrub.max = Math.max(0, state.events.length - 1);
    elements.empty.hidden = true;
    renderGraph();
    renderFrame();
  }

  async function loadFixture(path, label) {
    try {
      const response = await fetch(path);
      if (!response.ok) {
        throw new Error(`fixture request failed: ${response.status}`);
      }
      load(await response.json(), label);
    } catch (error) {
      showError(error);
    }
  }

  function pause() {
    if (state.timer) clearTimeout(state.timer);
    state.timer = null;
    elements.play.textContent = "play";
  }

  function tick() {
    if (state.index + 1 >= state.events.length) return pause();
    state.index += 1;
    renderFrame();
    state.timer = setTimeout(tick, 1000 / Number(elements.speed.value));
  }

  function play() {
    if (!state.events.length) return;
    if (state.timer) return pause();
    if (state.index + 1 >= state.events.length) state.index = 0;
    elements.play.textContent = "pause";
    state.timer = setTimeout(tick, 1000 / Number(elements.speed.value));
    renderFrame();
  }

  if (!d3) {
    showError("D3 v7 failed to load; check the CDN connection.");
    return;
  }
  elements["load-tiny"].onclick = () =>
    loadFixture("fixtures/tiny_step.json", "tiny fixture");
  elements["load-full"].onclick = () =>
    loadFixture("fixtures/step.json", "full fixture");
  elements.file.onchange = async (event) => {
    try {
      const file = event.target.files[0];
      if (file) load(JSON.parse(await file.text()), file.name);
    } catch (error) {
      showError(error);
    }
  };
  elements["first-sequence"].onchange = () => {
    if (state.raw.length) {
      try {
        load(state.raw, state.label);
      } catch (error) {
        showError(error);
      }
    }
  };
  elements.play.onclick = play;
  elements.previous.onclick = () => {
    pause();
    state.index = Math.max(0, state.index - 1);
    renderFrame();
  };
  elements.next.onclick = () => {
    pause();
    state.index = Math.min(state.events.length - 1, state.index + 1);
    renderFrame();
  };
  elements.scrub.oninput = () => {
    pause();
    state.index = Number(elements.scrub.value);
    renderFrame();
  };

  if (location.protocol === "file:") {
    elements.connection.textContent = "local file - choose JSON";
  } else {
    loadFixture("fixtures/tiny_step.json", "tiny fixture");
  }
}

if (typeof window !== "undefined" && typeof document !== "undefined") {
  startBrowser();
}
