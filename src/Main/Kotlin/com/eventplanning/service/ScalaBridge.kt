package com.eventplanning.service

import com.eventplanning.domain.Event
import com.eventplanning.domain.Venue
import java.time.LocalDateTime
import java.time.Duration
import java.util.ArrayList

object ScalaBridge {

    /**
     *scala class names to be used via reflection
     */
    private const val SLOT_FINDER_CLASS = "com.eventplanning.scheduling.SlotFinder"
    private const val EVENT_SCHEDULER_CLASS = "com.eventplanning.scheduling.EventScheduler"

    /**
     * Represents a specific time slot found for a venue.
     */
    data class VenueTimeSlot(
        val venue: Venue,
        val startTime: LocalDateTime,
        val freeUntil: LocalDateTime
    )

    /**
     * Result of the slot finding operation.
     * either a success with a list of VenueTimeSlot or an error message.
     */
    sealed class SlotFinderResult {
        data class Success(val slots: List<VenueTimeSlot>) : SlotFinderResult()
        data class Error(val message: String) : SlotFinderResult()
    }

    /**
     * Result of the scheduling operation.
     * either a success with a list of ScheduleEntry or an error message with failed count.
     */
    sealed class SchedulerResult {
        data class Success(val schedule: List<ScheduleEntry>) : SchedulerResult()
        data class Error(val message: String, val failedCount: Int) : SchedulerResult()
    }

    /**
     * simple data class representing a scheduled event entry.
     */
    data class ScheduleEntry(
        val eventId: String,
        val eventTitle: String,
        val venue: String,
        val dateTime: String
    )

    /**
     * Finds available venues and their specific time slots.
     * list of venues and events from the system.
     * required capacity for the venue.
     * desired date and time to check availability.
     * duration of the event to be scheduled.
     */
    fun findAvailableVenues(
        venues: List<Venue>,
        events: List<Event>,
        requiredCapacity: Int,
        dateTime: LocalDateTime,
        duration: Duration
    ): SlotFinderResult {

        /**
         * using reflection to call the Scala SlotFinder method
         * SLOT_FINDER_CLASS finds the scala class .slotFinder
         * method "findAllAvailableSlots" with the appropriate parameters.
         */
        return try {
            val clazz = Class.forName(SLOT_FINDER_CLASS)
            val method = clazz.getMethod(
                "findAllAvailableSlots",
                java.util.List::class.java,
                java.util.List::class.java,
                Int::class.javaPrimitiveType,
                LocalDateTime::class.java,
                Duration::class.java
            )

            /**
             * method invoke is null because it's a static method in Scala object
             * this runs the scala code
             * SlotFinder.findAllAvailableSlots(venues, events, requiredCapacity, dateTime, duration)
             */
            @Suppress("UNCHECKED_CAST")
            val rawResult = method.invoke(null, venues, events, requiredCapacity, dateTime, duration)
                    as List<Map<String, Any>>

            /**
             * converting raw scala result into Kotlin VenueTimeSlot objects
             * wraps the result in SlotFinderResult.Success
             */
            val slots = rawResult.map { map ->
                VenueTimeSlot(
                    venue = map["venue"] as Venue,
                    startTime = map["start"] as LocalDateTime,
                    freeUntil = map["end"] as LocalDateTime
                )
            }

            SlotFinderResult.Success(slots)

            /**
             * if the scala class is not found or any other exception occurs
             * return error instead of crashing the application/ui
             */
        } catch (e: ClassNotFoundException) {
            SlotFinderResult.Error("Scala SlotFinder not found.")
        } catch (e: Exception) {
            val cause = e.cause?.message ?: e.message ?: "Unknown error"
            SlotFinderResult.Error("SlotFinder error: $cause")
        }
    }

    /**
     * this is used by MainWindow.generateSchedule when the user clicks the generate schedule button
     */
    fun generateSchedule(events: List<Event>, venues: List<Venue>): SchedulerResult {
        return try {

            /**
             * schedulerClass finds the scala class .eventScheduler
             * resultClass finds the inner class EventScheduler$ScheduleResult
             * scheduleMethod gets the method "scheduleEventsList" with the appropriate parameters.
             */
            val schedulerClass = Class.forName(EVENT_SCHEDULER_CLASS)
            val resultClass = Class.forName("$EVENT_SCHEDULER_CLASS\$ScheduleResult")
            val scheduleMethod = schedulerClass.getMethod("scheduleEvents", java.util.List::class.java, java.util.List::class.java)

            /**
             * runs the scala algorithm
             * returns a scala object of type EventScheduler.ScheduleResult
             *  which contains the scheduling result
             *  converts the scheduleResult to a Map for easier parsing
             *  using parseSchedulerResult to convert the map into SchedulerResult
             */
            val result = scheduleMethod.invoke(null, events, venues)
            val mapMethod = schedulerClass.getMethod("scheduleToMap", resultClass)
            @Suppress("UNCHECKED_CAST")
            val resultMap = mapMethod.invoke(null, result) as Map<String, Any>
            parseSchedulerResult(resultMap)

            /**
             * if the scala class is not found or any other exception occurs
             */
        } catch (e: Exception) {
            val cause = e.cause?.message ?: e.message ?: "Unknown error"
            SchedulerResult.Error("Scheduler error: $cause", 0)
        }
    }

    /**
     * reads the success flag from the result map
     * if success is true, parses the schedule list into ScheduleEntry objects
     * extracts eventId, eventTitle, venue, and dateTime for each entry
     * wraps the list in SchedulerResult.Success and return SchedulerResult.Success(entries)
     * if success is false, extracts the error message and failed count
     * return SchedulerResult.Error(message, failedCount)
     */
    @Suppress("UNCHECKED_CAST")
    private fun parseSchedulerResult(resultMap: Map<String, Any>): SchedulerResult {
        val success = resultMap["success"] as? Boolean ?: false
        return if (success) {
            val scheduleList = resultMap["schedule"] as? ArrayList<Map<String, Any>>
                ?: return SchedulerResult.Error("Invalid schedule format", 0)
            val entries = scheduleList.map { entry ->
                ScheduleEntry(
                    eventId = entry["eventId"]?.toString() ?: "",
                    eventTitle = entry["eventTitle"]?.toString() ?: "",
                    venue = entry["venue"]?.toString() ?: "",
                    dateTime = entry["dateTime"]?.toString() ?: ""
                )
            }
            SchedulerResult.Success(entries)
        } else {
            val message = resultMap["message"] as? String ?: "Unknown scheduling error"
            val failedCount = (resultMap["failedCount"] as? Int) ?: 0
            SchedulerResult.Error(message, failedCount)
        }
    }
}