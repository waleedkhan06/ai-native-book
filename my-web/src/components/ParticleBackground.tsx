import React, { useCallback } from "react";
import Particles from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";
import type { Container, Engine } from "@tsparticles/engine";

const ParticleBackground: React.FC = () => {
  const particlesInit = useCallback(async (engine: Engine) => {
    await loadSlim(engine);
  }, []);

  const particlesLoaded = useCallback(async (container?: Container) => {
    // Optional: console.log("Particles loaded", container);
  }, []);

  return (
    <Particles
      id="tsparticles"
      init={particlesInit}
      loaded={particlesLoaded}
      options={{
        background: {
          color: {
            value: "transparent", // Let page background show through
          },
        },
        fpsLimit: 120,
        particles: {
          color: {
            value: ["#9333EA", "#A855F7", "#C4B5FD", "#FFFFFF"],
          },
          links: {
            enable: false, // Remove lines for cleaner look
          },
          move: {
            enable: true,
            speed: 0.5,
            direction: "none",
            random: true,
            straight: false,
            outModes: "out",
          },
          number: {
            density: {
              enable: true,
              area: 800,
            },
            value: 40, // Fewer particles = more subtle
          },
          opacity: {
            value: { min: 0.1, max: 0.5 },
          },
          shape: {
            type: "circle",
          },
          size: {
            value: { min: 1, max: 3 },
          },
        },
        detectRetina: true,
      }}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        zIndex: -1,
      }}
    />
  );
};

export default ParticleBackground;