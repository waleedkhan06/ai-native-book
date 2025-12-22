import { defineConfig } from "better-auth";
import { postgres } from "@better-auth/postgres-adapter";
import { drizzle } from "@better-auth/drizzle-adapter";
import { db } from "./db"; // Assuming you have a db connection

export const auth = defineConfig({
  secret: process.env.AUTH_SECRET || "your-development-secret",
  database: postgres({
    url: process.env.DATABASE_URL || "",
  }),
  socialProviders: {
    // Add social providers if needed
  },
  advanced: {
    // Enable email verification if needed
    emailVerification: true,
  },
  user: {
    // Define additional user fields for background questions
    additionalFields: {
      expertiseLevel: {
        type: "string",
        required: false,
        defaultValue: "beginner",
      },
      technicalBackground: {
        type: "string",
        required: true,
      },
      roboticsExperience: {
        type: "string",
        required: true,
      },
      primaryGoal: {
        type: "string",
        required: true,
      },
      hardwareAccess: {
        type: "string",
        required: true,
      },
      timeCommitment: {
        type: "string",
        required: true,
      },
      currentModule: {
        type: "number",
        required: false,
        defaultValue: 1,
      },
      currentChapter: {
        type: "string",
        required: false,
        defaultValue: "chapter-1.1",
      },
      progress: {
        type: "json",
        required: false,
        defaultValue: [],
      },
      preferredLanguage: {
        type: "string",
        required: false,
        defaultValue: "en",
      },
      learningPath: {
        type: "string",
        required: false,
        defaultValue: "general",
      },
    },
  },
  account: {
    // Account related configurations
  },
  session: {
    expiresIn: 7 * 24 * 60 * 60, // 7 days
  },
});