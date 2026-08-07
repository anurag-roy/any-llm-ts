import { AnyLLM } from "../src/index.js";

const llm = AnyLLM.createOpenAICompatible({
  name: "local",
  apiBase: "http://localhost:8000/v1",
  requiresApiKey: false,
});

const response = await llm.completion({
  model: "local-model",
  messages: [{ role: "user", content: "Hello" }],
});

console.log(response.choices[0]?.message.content);
