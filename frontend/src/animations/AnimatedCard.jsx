import { motion } from "framer-motion";

export default function AnimatedCard({ children }) {
  return (
    <motion.div
      whileHover={{
        y: -6,
        scale: 1.01,
      }}
      transition={{
        type: "spring",
        stiffness: 250,
        damping: 18,
      }}
    >
      {children}
    </motion.div>
  );
}