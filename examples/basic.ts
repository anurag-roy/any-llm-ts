import { completion } from "../src/index.js";

const response = await completion({
  provider: "openai",
  model: "gpt-4.1-mini",
  messages: [{ role: "user", content: "Say hello in one sentence." }],
});

console.log(response.choices[0]?.message.content);
