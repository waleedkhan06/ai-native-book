import type { ReactNode } from "react"
import clsx from "clsx"
import Link from "@docusaurus/Link";
import { FaRobot, FaBook, FaCode, FaTools, FaCogs, FaUnity, FaMicrochip, FaEye } from "react-icons/fa"

type CardItem = {
  title: string
  description: ReactNode
  icon: ReactNode
  path: string
  color: string
  showButton: boolean
}

const CardList: CardItem[] = [
  {
    title: "AI-Powered Learning",
    icon: <FaRobot size={40} color="#6D28D9" />,
    description: <>Interactive AI tutors and personalized learning paths to help you master robotics concepts.</>,
    path: "/docs/intro",
    color: "#8B5CF6",
    showButton: false,
  },
  {
    title: "Learning Path",
    icon: <FaBook size={40} color="#6D28D9" />,
    description: <>Complete coverage from fundamentals to advanced topics in physical AI and humanoid robotics.</>,
    path: "/docs/intro",
    color: "#8B5CF6",
    showButton: false,
  },
  {
    title: "Hands-on Projects",
    icon: <FaCode size={40} color="#6D28D9" />,
    description: <>Practical exercises and real-world projects to apply theoretical knowledge in simulation.</>,
    path: "/docs/intro",
    color: "#8B5CF6",
    showButton: false,
  },
  {
    title: "Advanced Tools",
    icon: <FaTools size={40} color="#6D28D9" />,
    description: <>Cutting-edge tools and frameworks to build and deploy your AI and robotics projects at scale.</>,
    path: "/docs/intro",
    color: "#8B5CF6",
    showButton: false,
  },
  {
    title: "ROS2 Fundamentals",
    icon: <FaCogs size={40} color="#6D28D9" />,
    description: (
      <>
        Core ROS2 concepts and architecture, covering nodes, topics, services, and actions, along with package
        management.
      </>
    ),
    path: "/docs/module-1",
    color: "#A78BFA",
    showButton: true,
  },
  {
    title: "Gazebo & Unity Simulation",
    icon: <FaUnity size={40} color="#6D28D9" />,
    description: (
      <>
        Physics-based robot simulation with Unity integration for advanced visuals, including environment modeling and
        testing.
      </>
    ),
    path: "/docs/module-2",
    color: "#A78BFA",
    showButton: true,
  },
  {
    title: "NVIDIA Isaac Platform",
    icon: <FaMicrochip size={40} color="#6D28D9" />,
    description: (
      <>
        GPU-accelerated robotics development with AI perception and navigation tools, enabling simulation to real-world
        deployment.
      </>
    ),
    path: "/docs/module-3",
    color: "#A78BFA",
    showButton: true,
  },
  {
    title: "Vision-Language-Action",
    icon: <FaEye size={40} color="#6D28D9" />,
    description: (
      <>
        Computer vision for robotics systems, multimodal AI integration frameworks, and action planning and execution.
      </>
    ),
    path: "/docs/module-4",
    color: "#A78BFA",
    showButton: true,
  },
]

function Card({ title, description, icon, path, color, showButton }: CardItem) {
  return (
    <div className={clsx("col col--3")}>
      <div className="feature-card">
        <div className="feature-icon" style={{ marginBottom: "1rem" }}>
          {icon}
        </div>
        <h3 className="card-title">{title}</h3>
        <div className="card-description">{description}</div>
        {showButton && (
          <div style={{ marginTop: "1.5rem" }}>
            <Link
              className="button button--primary button--sm"
              style={{
                backgroundColor: color,
                borderColor: color,
                padding: "0.5rem 1rem",
                borderRadius: "8px",
                color: "white",
                textDecoration: "none",
                display: "inline-block",
              }}
              href={path}
            >
              Explore Module
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}

export default function CardComponent(): ReactNode {
  return (
    <section className="features">
      <div className="container">
        <div className="row">
          {CardList.map((props, idx) => (
            <Card key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  )
}
