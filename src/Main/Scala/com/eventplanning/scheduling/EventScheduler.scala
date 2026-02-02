package com.eventplanning.scheduling

import com.eventplanning.domain.Event
import com.eventplanning.domain.Venue
import java.time.LocalDateTime
import java.time.LocalTime
import scala.annotation.tailrec
import scala.jdk.CollectionConverters._

object EventScheduler {

  private case class ScheduledEvent(
                             event: Event,
                             assignedVenue: Venue,
                             assignedDateTime: LocalDateTime
                           )

  /**
   * outcome of the scheduling process
   * either Success with the scheduled events
   * or Failure with a message and list of problematic events
   */
  sealed trait ScheduleResult
  private case class Success(schedule: List[ScheduledEvent]) extends ScheduleResult
  private case class Failure(message: String, problematicEvents: List[Event]) extends ScheduleResult

  /**
   * We set it so it looks for available slots within the next 7 days
   * in 30-minute increments
   */
  private val SEARCH_DAYS = 7
  private val SLOT_MINUTES = 30
  private val MAX_SCHEDULING_ATTEMPTS: Int =
    SEARCH_DAYS * 24 * (60 / SLOT_MINUTES)

  private val WORK_DAY_START = LocalTime.of(8, 0)
  private val WORK_DAY_END = LocalTime.of(20, 0)

  /**
   *this is what ScalaBridge calls
   *param events: list of events to schedule
   *param venues: list of available venues
   *return: ScheduleResult indicating success or failure
   */
  def scheduleEvents(
                      events: java.util.List[Event],
                      venues: java.util.List[Venue]
                    ): ScheduleResult = {

    /**
     * basic null and empty checks
     */
    if (events == null || venues == null) return Failure("Input lists cannot be null", List.empty)
    /**
     * convert Java event and venue lists to Scala lists
     */
    val eventList = events.asScala.toList
    val venueList = venues.asScala.toList

    /**
     * if there are no events, return empty schedule
     * but if there are no venues, return failure
     */
    if (eventList.isEmpty) return Success(List.empty)
    if (venueList.isEmpty) return Failure("No venues available to schedule events.", eventList)

    /**
     * sort events by duration descending to try to fit longer events first
     */
    val sortedEvents = eventList.sortBy(_.getDuration).reverse

    /**
     * start the recursive scheduling process
     * remainingEvents: sortedEvents
     * availableVenues: venueList
     * scheduledSoFar: empty list
     */
    scheduleEventsRecursive(sortedEvents, venueList, List.empty)
  }

  /**
   *tail-recursive method to schedule events one by one
   * remainingEvents: list of events yet to be scheduled
   * availableVenues: list of venues to choose from
   * scheduledSoFar: list of successfully scheduled events
   */
  @tailrec
  private def scheduleEventsRecursive(
                                       remainingEvents: List[Event],
                                       availableVenues: List[Venue],
                                       scheduledSoFar: List[ScheduledEvent]
                                     ): ScheduleResult = {
    /**
     * base case: if no remaining events, return success with scheduled events
     */
    remainingEvents match {
      case Nil =>
        Success(scheduledSoFar.reverse)

      /**
       * recursive case: try to schedule the next event
       */
      case event :: restEvents =>
        findBestSlotForEvent(event, availableVenues, scheduledSoFar) match {
          case Some(scheduledEvent) =>
            val updated = scheduledEvent :: scheduledSoFar
            scheduleEventsRecursive(restEvents, availableVenues, updated)

          /**
           * could not find a slot for this event
           * return failure with message and remaining events
           */
          case None =>
            Failure(
              s"Conflict: '${event.getTitle}' (Needs Cap: ${event.getMaxParticipants}) could not be booked.\n" +
                s"Could not find any free slot starting from ${event.getDateTime} within limits.",
              event :: restEvents
            )
        }
    }
  }

  /**
   *tries to find the best venue and time slot for the given event
   * filters venues by capacity and sorts by smallest capacity first
   * try to find a free slot in each venue
   */
  private def findBestSlotForEvent(
                                    event: Event,
                                    venues: List[Venue],
                                    existingSchedule: List[ScheduledEvent]
                                  ): Option[ScheduledEvent] = {

    /**
     * makes sure it can hold the event's participants
     */
    val capableVenues = venues
      .filter(_.getCapacity >= event.getMaxParticipants)
      .sortBy(_.getCapacity)

/**
 * looks for a free slot one by one in the capable venues
 */
    findSlotInVenues(event, capableVenues, existingSchedule)
  }

  /**
   *find a free slot for the event in the list of venues
   * tries each venue in order until a slot is found or all venues exhausted
   */
  @tailrec
  private def findSlotInVenues(
                                event: Event,
                                venues: List[Venue],
                                schedule: List[ScheduledEvent]
                              ): Option[ScheduledEvent] = {
    venues match {

      /**
       * no more venues to try
       * return None
       */
      case Nil => None
      case venue :: tail =>
        findNextTimeSlot(event, venue, event.getDateTime, schedule, 0) match {

          /**
           * found a time slot in this venue
           * return the scheduled event
           */
          case Some(time) => Some(ScheduledEvent(event, venue, time))
          case None => findSlotInVenues(event, tail, schedule)
        }
    }
  }

  /**
   *recursively looks for the next available time slot for the event in the venue
   * starts from candidateTime and checks for conflicts
   */
  @tailrec
  private def findNextTimeSlot(
                                event: Event,
                                venue: Venue,
                                candidateTime: LocalDateTime,
                                schedule: List[ScheduledEvent],
                                attempts: Int
                              ): Option[LocalDateTime] = {

    /**
     * stop if exceeded max attempts
     * (7 days worth of 30-minute slots)
     */
    if (attempts > MAX_SCHEDULING_ATTEMPTS) {
      None
    }

    /**
     * stop if outside working hours 8pm - move to next day morning
     */
    else if (candidateTime.toLocalTime.isAfter(WORK_DAY_END.minus(event.getDuration))) {

      val nextMorning = candidateTime.plusDays(1).withHour(WORK_DAY_START.getHour).withMinute(0)
      findNextTimeSlot(event, venue, nextMorning, schedule, attempts + 1)

      /**
       * if no conflict, return this candidate time
       */
    } else if (!hasConflict(candidateTime, event.getDuration, venue, schedule)) {

      Some(candidateTime)
    } else {
      /**
       * conflict found, try next 30-minute slot
       */
      findNextTimeSlot(event, venue, candidateTime.plusMinutes(30), schedule, attempts + 1)
    }
  }

  /**
   *checks if the proposed time slot conflicts with existing scheduled events in the venue
   */
  private def hasConflict(
                           start: LocalDateTime,
                           duration: java.time.Duration,
                           venue: Venue,
                           existingSchedule: List[ScheduledEvent]
                         ): Boolean = {
    val end = start.plus(duration)

    existingSchedule.exists { scheduled =>

      /**
       * only events in the same venue can conflict
       * check for time overlap
       */
    scheduled.assignedVenue.getId == venue.getId &&
        timeOverlap(start, end, scheduled.assignedDateTime, scheduled.assignedDateTime.plus(scheduled.event.getDuration))
    }
  }

  /**
   *return true if the two time intervals overlap
   * [start1, end1] and [start2, end2]
   */
  private def timeOverlap(
                           start1: LocalDateTime,
                           end1: LocalDateTime,
                           start2: LocalDateTime,
                           end2: LocalDateTime
                         ): Boolean = {
    start1.isBefore(end2) && end1.isAfter(start2)
  }

  /**
   *converts ScheduleResult to a Java Map for interoperability
   * ScalaBridge can easily read and create SchedulerResult,
   * but to return to Java code, we convert it to a Map
   */
  def scheduleToMap(result: ScheduleResult): java.util.Map[String, Object] = {
    val map = new java.util.HashMap[String, Object]()

    result match {

      /**
       * on success, put success=true and the schedule details
       * each scheduled event includes eventId, eventTitle, venue, dateTime
       */
      case Success(schedule) =>
        map.put("success", java.lang.Boolean.TRUE)
        val scheduleList = new java.util.ArrayList[java.util.Map[String, Object]]()
        schedule.foreach { scheduled =>
          val eventMap = new java.util.HashMap[String, Object]()
          eventMap.put("eventId", scheduled.event.getId)
          eventMap.put("eventTitle", scheduled.event.getTitle)
          eventMap.put("venue", scheduled.assignedVenue.getName)
          eventMap.put("dateTime", scheduled.assignedDateTime.toString)
          scheduleList.add(eventMap)
        }
        map.put("schedule", scheduleList)

      /**
       * on failure, put success=false, message, and count of failed events
       */
      case Failure(message, problematicEvents) =>
        map.put("success", java.lang.Boolean.FALSE)
        map.put("message", message)
        map.put("failedCount", Integer.valueOf(problematicEvents.size))
    }
    map
  }
}